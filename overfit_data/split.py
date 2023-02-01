import os
import pathlib
import shutil

id = "02691156"
IfSplitByTheNumberOfSamplings=False
IfSplitToTrainValueTest=True

mask_dir = pathlib.Path(f"mask/{id}/")
images_dir = pathlib.Path(f"sampled/{id}/")

masks = sorted(list(mask_dir.glob("*.png")))

n_train = int(len(masks) * 0.8)
n_val = int(len(masks) * 0.1)
n_test = len(masks) - n_train - n_val


train_path = pathlib.Path(f"sampled/train/{id}/")
val_path = pathlib.Path(f"sampled/val/{id}/")
test_path = pathlib.Path(f"sampled/test/{id}/")

train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)


tmask = masks[:n_train]
vmask = masks[n_train : n_train + n_val]
temask = masks[n_train + n_val :]

if IfSplitToTrainValueTest:
    for image in tmask:
        for im in images_dir.glob("*" + image.name):
            shutil.copy(str(im), str(train_path))

    for image in vmask:
        for im in images_dir.glob("*" + image.name):
            shutil.copy(str(im), str(val_path))

    for image in temask:
        for im in images_dir.glob("*" + image.name):
            shutil.copy(str(im), str(test_path))

elif IfSplitByTheNumberOfSamplings:
    for num_sampling in range(14):
        splited_train_path=pathlib.Path(f"overfit_data/sampled/splited_test/{id}/{num_sampling+1}")
        splited_train_path.mkdir(parents=True,exist_ok=True)
    for image in temask:
        for im in images_dir.glob("*" + image.name):
            _, num_sampling, mask_name=im.name.split("_",maxsplit=3)
            shutil.copy(str(im),(f"overfit_data/sampled/splited_test/{id}"+f"/{num_sampling}"))


