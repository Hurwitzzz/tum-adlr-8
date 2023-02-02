import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# img should be a tensor
def EnlargePointsForVisualization (img):
    convert_PIL=transforms.ToPILImage()
    
    rows=(img==1).nonzero(as_tuple=True)[0].tolist()
    cols=(img==1).nonzero(as_tuple=True)[1].tolist()
    # print(rows,cols)
    img_clone=img.clone()
    for row,col in zip(rows,cols): #for each sampling points
        for i in range(-2,3):
            for j in range(-2,3):
                if((row+i<=99 and row+i>=0)and (col+j<=99 and col+j>=0)):
                    img_clone[row+i][col+j]=1
    return convert_PIL(img_clone)


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def plot_example_imgs_from_dataset(dataset,num_example_imgs):

    plt.figure(figsize=(20, 10 * num_example_imgs))
    for i in range(num_example_imgs):
        
        img=dataset[i]['image']
        mask=dataset[i]['mask']
        print(i)
        # sampled imgs
        plt.subplot(num_example_imgs, 2, i * 2 + 1)
        plt.imshow(img.numpy().transpose(1,2,0))
        plt.title(dataset.GetFileName(i))
        # plt.axis('off')
        # if i == 0:
        #     plt.title("Sampled image")
        
        # mask
        plt.subplot(num_example_imgs, 2, i * 2 + 2)
        plt.imshow(mask.numpy())
        plt.title(dataset.GetFileName(i))
        # plt.axis('off')
        # if i == 0:
        #     plt.title("Mask image")
    plt.show()

#img/mask/truth is [num,H,W] array
def plot_img_predictedmask_and_truth(img,mask,truth):
    if img.shape[0]!=mask.shape[0]:
        return "The 1st dimension of img and mask should be same"
    else:
        num=img.shape[0]
        plt.figure(figsize=(30, 10 * num))
        for i in range(num):
            # sampled imgs
            plt.subplot(num, 3, i * 3 + 1)
            plt.imshow(img[i])
            plt.axis('off')
            if i == 0:
                plt.title("Sampled image")
            
            # predicted mask
            plt.subplot(num, 3, i * 3 + 2)
            plt.imshow(mask[i])
            plt.axis('off')
            if i == 0:
                plt.title("Predicted mask")
            # ground truth mask
            plt.subplot(num, 3, i * 3 + 3)
            plt.imshow(truth[i])
            plt.axis('off')
            if i == 0:
                plt.title("Ground truth mask")
        plt.show()
