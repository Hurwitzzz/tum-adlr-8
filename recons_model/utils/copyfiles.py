import os
import shutil

# source_folder = r"/home/hewei/Downloads/mask/02691156/"
# destination_folder = r"/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/truemask/02691156/"



# Use this file to copy mask to our project folder and copy each mask 10 times to match the number of sample-files
def CopyAndMultiply(source_folder, destination_folder, Mul):
    # fetch all files
    for file_name in os.listdir(source_folder):
        # construct full file path
        for i in range(Mul):
            source = source_folder + file_name
            destination = destination_folder + str(i)+'_'+file_name
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                # print('copied', file_name)




# To generate reduced dataset:
def ReduceData(source_folder, dst_folder, num):
    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(source_folder, x)),
                        os.listdir(source_folder) ) )
    for file_name in list_of_files[0:num]:
            source=source_folder+file_name
            destination=dst_folder+file_name
            # copy files
            if os.path.isfile(source):
                shutil.copy(source,destination)


source_mask= r"/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/truemask/02691156/"
dst_mask=r"/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/Mask_reduced/02691156/"

source_sampled= r"/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/sampled/02691156/"
dst_sampled=r"/home/hewei/TUM/ADLR/tum-adlr-8/UNet-Pytorch/data/Sampled_reduced/02691156/"

num=1000
ReduceData(source_mask,dst_mask,num)
ReduceData(source_sampled,dst_sampled,num)