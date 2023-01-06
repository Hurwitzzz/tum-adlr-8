import os
import pathlib
import shutil

id = "534534543"

mask_dir = pathlib.Path(f"./mask/{id}/")
images_dir = pathlib.Path(f"./sampled/{id}/")

masks = sorted(list(mask_dir.glob("*.png")))
images = list(images_dir.glob("*.png"))

n_train = int(len(masks) * 0.8)
n_val = int(len(masks) * 0.1)
n_test = len(masks) - n_train - n_val


train_path = pathlib.Path(f"./sampled/train/{id}/")
val_path = pathlib.Path(f"./sampled/val/{id}/")
test_path = pathlib.Path(f"./sampled/test/{id}/")

train_path.mkdir(parents=True, exist_ok=True)
val_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)


tmask = masks[:n_train]
vmask = masks[n_train : n_train + n_val]
temask = masks[n_train + n_val :]

for image in tmask:
    for im in images_dir.glob("*" + image.name):
        shutil.move(str(im), str(train_path))

for image in vmask:
    for im in images_dir.glob("*" + image.name):
        shutil.move(str(im), str(train_path))

for image in temask:
    for im in images_dir.glob("*" + image.name):
        shutil.move(str(im), str(train_path))
