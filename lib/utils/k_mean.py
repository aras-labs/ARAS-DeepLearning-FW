import torch
import numpy as np
from lib.engine.kmeans_pytorch import kmeans
import matplotlib.pyplot as plt
# data
data_size, dims, num_clusters = 1000, 4, 3
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)

from lib.data.datasets import cihp
from lib.system.identification import merge_system_environment_to_cfg, set_gpu_environment
from core.config import cfg
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchmetrics import IoU
import torch.nn.functional as F
from lib.data.transforms import transforms


def get_output_img(msk, seg, inst_num, coord):

    imgshape = msk.shape
    msk_img = np.argmax(msk,-1)
    seg_img = np.argmax(seg,-1)
    inst_num = np.rint(inst_num)

    instance_img = np.zeros(imgshape,dtype=np.uint8)
    non0 = np.nonzero(msk_img)
    coord_map = coord[non0[0],non0[1]]
    coord_map = coord_map[:2]
    indices = np.transpose(non0)
    final, data, clusters = finalclustering(coord_map, indices, inst_num, 0.5)
    instance_img[final[0],final[1]] = final[2]
    cmap = color_map()

    msk_img[msk_img==1] = 255
    msk_img = np.uint8(msk_img)
    seg_img = cmap[seg_img]
    instance_img = cmap[instance_img]

    return msk_img, seg_img, instance_img


cfg.merge_from_file('./config/train_parameter.yml')
cfg = merge_system_environment_to_cfg(cfg)
cfg.freeze()

transform = transforms.ToTensor()

DATASET_PATH = cfg.DATASET.PATH
train_ds = cihp.CIHP(DATASET_PATH, train=True, transform=transform)
val_ds = cihp.CIHP(DATASET_PATH, train=False, transform=transform)
i=1530
num_clusters = val_ds[i][1]['num_ins'].numpy()+1
x = val_ds[i][1]['h_cor'].to(torch.device('cuda:0')).permute(1,2,0).reshape(-1,4)


cl, c = kmeans(
        X=x,
        num_clusters=num_clusters,
        # distance='cosine',
        cluster_centers=[],
        tol=0,
        tqdm_flag=False,
        iter_limit=100,
        device=torch.device('cuda'),
        gamma_for_soft_dtw=0.1)

plt.imshow(cl.reshape(-1, 512))
plt.show()
plt.imshow(val_ds[i][1]['h_ins'])
plt.show()
plt.imshow(val_ds[i][1]['h_cor'][0])
plt.show()
plt.imshow(val_ds[i][1]['h_cor'][1])
plt.show()
plt.imshow(val_ds[i][1]['h_cor'][2])
plt.show()
plt.imshow(val_ds[i][1]['h_cor'][3])
plt.show()