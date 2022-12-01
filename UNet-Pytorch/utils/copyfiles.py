# Use this file to copy mask to our project folder and copy each mask 10 times to match the number of sample-files

import os
import shutil

source_folder = r"/home/hewei/Downloads/mask/02691156/"
destination_folder = r"/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/truemask/02691156/"


# fetch all files
for file_name in os.listdir(source_folder):
    # construct full file path
    for i in range(10):
        source = source_folder + file_name
        destination = destination_folder + str(i)+'_'+file_name
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            # print('copied', file_name)
