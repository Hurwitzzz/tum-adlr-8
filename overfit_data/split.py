
import os
import pathlib
import shutil

id = "534534543"

mask_dir = pathlib.Path(f"./masks/{id}/")
images_dir =  pathlib.Path(f"./samples/{id}/")

masks = sorted(list(mask_dir.glob('*.png')))
images = list(images_dir.glob('*.png'))

n_train = int(len(masks) * 0.8)
n_val = int(len(masks) * 0.1)
n_test = len(masks) - n_train - n_val


train_path = pathlib.Path(f"./samples/train/{id}/")
val_path = pathlib.Path(f"./samples/val/{id}/")
test_path = pathlib.Path(f"./samples/test/{id}/")

train_path.mkdir(exist_ok=True)
val_path.mkdir(exist_ok=True)
test_path.mkdir(exist_ok=True)



tmask = masks[:n_train]
vmask = masks[n_train:n_train+n_val]
temask = masks[n_train+n_val:]

for image in tmask:
    for im in images_dir.glob("*" + image.name):
        shutil.move(im, train_path)
        
for image in vmask:
    for im in images_dir.glob("*" + image.name):
        shutil.move(im, train_path)

for image in temask:
    for im in images_dir.glob("*" + image.name):
        shutil.move(im, train_path)