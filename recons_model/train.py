import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


from evaluate import evaluate, predict
from unet import UNet
from utils.data_loading import Tactile2dDataset
from utils.dice_score import dice_loss
from utils.utils import plot_example_imgs_from_dataset


IFTEST=True

train_dir_img = Path("overfit_data/sampled/train/02691156/")
test_dir_img = Path("overfit_data/sampled/test/02691156/")
val_dir_img = Path("overfit_data/sampled/val/02691156/")

dir_mask = Path("overfit_data/mask/02691156")
dir_checkpoint = Path("checkpoints")
test_set = None



def train_net(
    net,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    test_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 1,
    amp: bool = False,
    iftest: bool = False
):
    if (iftest == False):
        # 1. Create dataset
        train_set = Tactile2dDataset(train_dir_img, dir_mask, img_scale)
        val_set = Tactile2dDataset(val_dir_img, dir_mask, img_scale)

        # 2. Split into train / validation partitions
        n_val = len(val_set)
        n_train = len(train_set)

        # # 2.1 have a look at the example imgs
        plot_example_imgs_from_dataset(train_set,4)

        eval_for_n = 500

        # 3. Create data loaders
        loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        # (Initialize logging)
        experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_percent=val_percent,
                save_checkpoint=save_checkpoint,
                img_scale=img_scale,
                amp=amp,
            )
        )

        logging.info(
            f"""Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        """
        )

        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss()
        global_step = 0

        # 5. Begin training
        for epoch in range(1, epochs + 1):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
                for batch in train_loader:
                    # batch is dict{'image':....., 'mask':....}
                    # The shape of image is [batch_size,1,100,100],
                    # that of mask is [batch_size,100,100]
                    images = batch["image"]
                    true_masks = batch["mask"]

                    assert images.shape[1] == net.n_channels, (
                        f"Network has been defined with {net.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        loss = criterion(masks_pred, true_masks) + dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True,
                        )

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({"train loss": loss.item(), "step": global_step, "epoch": epoch})
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # Evaluation round
                    if (global_step % eval_for_n) == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace("/", ".")
                            if not torch.isinf(value).any():
                                histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms["Gradients/" + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, (val_image, val_mask_pred, val_mask_true) = evaluate(net, val_loader, device)
                        train_score, (_, _, _) = evaluate(net, train_loader, device)

                        logging.info("Validation Dice score: {}".format(val_score))
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation Dice": val_score,
                                "train Dice": train_score,
                                "train_images": wandb.Image(images[0].cpu()),
                                "train_masks": {
                                    "true": wandb.Image(true_masks[0].float().cpu()),
                                    "pred": wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                "val_images": wandb.Image(val_image[0].cpu()),
                                "val_masks": {
                                    "true": wandb.Image(val_mask_true[0].float().cpu()),
                                    "pred": wandb.Image(val_mask_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                "step": global_step,
                                "epoch": epoch,
                                **histograms,
                            }
                        )

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch)))
                logging.info(f"Checkpoint {epoch} saved!")
    
    # Test mode
    else:
        test_set = Tactile2dDataset(test_dir_img, dir_mask, img_scale)
        loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

        # (Initialize logging)
        experiment = wandb.init(project="U-Net", resume="allow", anonymous="allow")
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_percent=val_percent,
                save_checkpoint=save_checkpoint,
                img_scale=img_scale,
                amp=amp,
            )
        )    
        val_score, (dice_score_set,val_image_set, val_mask_pred_set, val_mask_true_set) = predict(net, test_loader, device)
        
        # num=8
        # col=3
        # for i in range(num):
        #     plt.subplot(num,col,i*col+1)
        #     plt.imshow(val_image[i][0].cpu())
        #     if i==0:
        #         plt.title("sampled points")
        #     plt.subplot(num,col,i*col+3)
        #     plt.imshow(val_mask_true.argmax(dim=1)[i].float().cpu())
        #     if i==0:
        #         plt.title("truth")
        #     plt.subplot(num,col,i*col+2)
        #     plt.imshow(val_mask_pred.argmax(dim=1)[i].float().cpu())
        #     if i==0:
        #         plt.title("prediction")       
        # plt.show()

        logging.info("Validation Dice score: {}".format(val_score))
        for i in range(len(val_image_set)):
            for j in range(val_image_set[i].size(dim=0)):
                experiment.log(
                    {
                        "test Dice": dice_score_set[i],
                        # "test_images": wandb.Image(val_image[0].cpu()),

                        "image": wandb.Image(val_image_set[i][j][0].cpu()),
                        "test_masks": {
                            "true": wandb.Image(val_mask_true_set[i].argmax(dim=1)[j].float().cpu()),
                            "pred": wandb.Image(val_mask_pred_set[i].argmax(dim=1)[j].float().cpu()),
                        },
                        "step": i*batch_size+j+1,
                    }
                )

        

def get_args():
    parser = argparse.ArgumentParser(description="Train the UNet on images and target masks")
    parser.add_argument("--epochs", "-e", metavar="E", type=int, default=2, help="Number of epochs")
    parser.add_argument(
        "--batch-size", "-b", dest="batch_size", metavar="B", type=int, default=32, help="Batch size"
    )  # it could be 32/64 on cloud
    parser.add_argument(
        "--learning-rate", "-l", metavar="LR", type=float, default=1e-5, help="Learning rate", dest="lr"
    )
    parser.add_argument("--load", "-f", type=str, default=False, help="Load model from a .pth file")
    parser.add_argument("--scale", "-s", type=float, default=1, help="Downscaling factor of the images")
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision")
    parser.add_argument("--bilinear", action="store_true", default=False, help="Use bilinear upsampling")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images, =1 for our input
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(
        f"Network:\n"
        f"\t{net.n_channels} input channels\n"
        f"\t{net.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")
    if IFTEST:
        net.load_state_dict(torch.load("checkpoints/INTERRUPTED.pth", map_location=device))
        logging.info(f"Model loaded from checkpoints/INTERRUPTED.pth")

    net.to(device=device)
    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            test_percent=0.1,
            amp=args.amp,
            iftest=IFTEST
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        raise
