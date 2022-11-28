import matplotlib.pyplot as plt


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