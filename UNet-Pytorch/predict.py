import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
<<<<<<< HEAD
import train
=======
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
<<<<<<< HEAD
from utils.utils import plot_img_predictedmask_and_truth

#predict single image
=======

>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
<<<<<<< HEAD
    img = img.unsqueeze(0) #img is a tensor, img.size()=[1,1,100,100]
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img) #tensor, size()=[1,n_class,H,W]
=======
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

<<<<<<< HEAD
def predict_tensorimg(net,tensor_img,device,scale_factor=1):
    net.eval()
    if tensor_img.dim==2:
        tensor_img=tensor_img[None,None,:,:]
    tensor_img = tensor_img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img) #tensor, size()=[1,n_class,H,W]
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]
        output = output.squeeze()
    if net.n_classes==2:
        return F.one_hot(output.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

#predict multi images
#num is the number of images in the folder that you want to predict
def predict_imgs_in_folder(net,input_folder,truth_folder,index,device,draw):
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(input_folder, x)),
                        os.listdir(input_folder) ) )
    num=index[1]-index[0]
    img=[None]*num
    truth=[None]*num #truth stores the ground truth masks of inputs
    for i,file_name in enumerate(list_of_files[index[0]:index[1]]):
        img[i] = Image.open(input_folder+file_name)
        truth[i]=Image.open(truth_folder+file_name)
        #initialize list to store each predicted mask
        if i==0:
            mask=np.empty([num,net.n_classes,img[i].size[0],img[i].size[1]])
        mask[i]=predict_img(net,img[i],device,1,0.5)
    #visualize predict result
    if draw:
        #change mask from [0,1] to [0,255] and reduce one dimension to [num, H,W]
        mask=mask_to_image(mask)
        img_array=np.asarray(img)
        truth_array=np.asarray(truth)
        plot_img_predictedmask_and_truth(img_array,mask,truth_array)
    return mask # mask is a [num,H,W] array

# predict val/test dataset
# index is a list [begin,end] which indicate the index of dataset. Doesn't include dataset[end]!
def predict_imgs_in_dataset(net,dataset,index,device,draw):
    num=index[1]-index[0]
    input=[None]*num
    truth=[None]*num
    pre_mask=[None]*num
    for i in range(index[0],index[1]):
        input[i]=dataset[i]['image']
        truth[i]=dataset[i]['mask']
        pre_mask[i]=predict_tensorimg(net,input[i],device) #return [n_classes, H,W] np.array
    if draw:
        pre_mask=mask_to_image(pre_mask)
        input_array=np.asarray(input)
        truth_array=np.asarray(truth)
        plot_img_predictedmask_and_truth(input_array,mask,truth_array)
    return pre_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch2.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', default='data/sampled/02691156/0_1807d521bc2164cd8c4f5e741c2f9cdb-6.png', metavar='INPUT', nargs='+', help='Filenames of input images', required=False)
    parser.add_argument('--output', '-o', default='output.img',metavar='OUTPUT', nargs='+', help='Filenames of output images')
=======

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
<<<<<<< HEAD
    parser.add_argument('--scale', '-s', type=float, default=1,
=======
    parser.add_argument('--scale', '-s', type=float, default=0.5,
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
<<<<<<< HEAD
    #class=0: black; class=1: white, means there is an object
    if mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255).astype(np.uint8))
    elif mask.ndim==4:
        return np.argmax(mask,axis=1)*255

        


if __name__ == '__main__':
    PREDICT_SINGLE_IMAGE=0
    PREDICT_MULTI_IMAGE=1
    PREDICT_DATASET=0

=======
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

<<<<<<< HEAD
    net = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
=======
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

<<<<<<< HEAD
    if PREDICT_SINGLE_IMAGE:
        #predict single image
        filename=in_files
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)
        mask=predict_img(net,img,args.scale,args.mask_threshold,device)
        if not args.no_save:
                out_filename = "OUT_"+filename
                result = mask_to_image(mask)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')
    elif PREDICT_MULTI_IMAGE:
        #predict multi images
        # input_folder="/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/Sampled_reduced/02691156/"
        # truth_folder="/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/Mask_reduced/02691156/"
        input_folder="/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/sampled/02691156/"
        truth_folder="/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/truemask/02691156/"
        index=[2000,2005]
        mask_mul=predict_imgs_in_folder(net,input_folder,truth_folder,index,device,draw=True) #mask_mul.shape=[num,H,W], array
    # elif PREDICT_DATASET:
    #     #predict dataset
    #     index=[0:5]
    #     pre_mask=predict_imgs_in_dataset(net,train.test_set,)
=======
    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
>>>>>>> 3f3aa2c1076536f4f700af99a47fbea39a25d094
