import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

def show_example(fig, ax, img, labels):
    for i in range(6):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].axis("off")
    # show the orginal image
    ax[0].imshow(img.permute(1, 2, 0))
    # show the mask of fg-bg
    ax[1].imshow(labels['mask'], cmap='gray')
    # show the semantic segmentation lables
    ax[2].imshow(labels['seg'], cmap='gray')
    # show the human instance lables
    ax[3].imshow(labels['h_ins'], cmap='gray')
    # show the first channel of cords
    ax[4].imshow(labels['h_cor'][0])
    print(labels['h_cor'][0].unique())
    # show all instances
    ax[5].imshow(labels['ins'], cmap='gray')
    print(labels['num_ins'])
    fig.show()
    plt.pause(.1)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def dataset_visualization(data):
    img = (data[0] * 255).to(torch.uint8)
    boxes = data[1]['boxes']
    colors = ["blue", "yellow"]
    ll = np.array(["bg", 'ins', "pupil"])
    print(type(list(ll[data[1]['labels'].numpy()])[0]), list(ll[data[1]['labels'].numpy()]))
    result = draw_bounding_boxes(img, boxes, width=5, colors=colors, labels=list(map(str, list(ll[data[1]['labels'].numpy()]))))

    show(result)