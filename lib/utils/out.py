import matplotlib.pyplot as plt
import torch

from core.config import cfg
from lib.data.transforms import transforms
from lib.data.datasets import cihp
from lib.system.identification import merge_system_environment_to_cfg


cfg.merge_from_file('./config/train_parameter.yml')
cfg = merge_system_environment_to_cfg(cfg)
cfg.freeze()

transform = transforms.ToTensor()

DATASET_PATH = cfg.DATASET.PATH
train_ds = cihp.CIHP(DATASET_PATH, train=True, transform=transform)
val_ds = cihp.CIHP(DATASET_PATH, train=False, transform=transform, imag_size=32)


import numpy as np


# def affinity(s, var):
#     n = s.shape[1]
#     A = torch.zeros((n, n))
#
#     # def sqrt_func(i, j):
#     #     if i == j:
#     #         res = 0
#     #     else:
#     #         p = s[:, i] - s[:, j]
#     #         res = torch.exp(-torch.sqrt(torch.sum(p * p)) / (2 * var))
#     #         print(n, i, j)
#     #     A[i, j] = res
#
#     # Parallel(n_jobs=8)(delayed(sqrt_func)(i, j) for i in range(n) for j in range(n))
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 res = 0
#             else:
#                 p = s[:, i] - s[:, j]
#                 res = torch.exp(-torch.sqrt(torch.sum(p*p)) / (2*var))
#             A[i, j] = res
#     return A
def affinity(s, non0_coord, var):
    n = s.shape[1]
    A = torch.tensor(range(n*n), dtype=torch.float)
    print(n, A.shape)
    def fn(x):
        i = int(x // n)
        j = int(x % n)
        if i == j:
            res = 0
        else:
            p = s[:, i] - s[:, j]
            p2 = non0_coord[:,i] - non0_coord[:,j]
            # print(p,p2)
            res = torch.exp(-torch.sqrt(torch.sum(p * p)) / (2 * var)) + torch.exp(-torch.sqrt(torch.sum(p2*p2)) / (2*var))
        return res
    A.apply_(lambda x: fn(x))
    A = A.reshape(n, n)
    print(A)
    # for i in range(n):
    #     for j in range(n):
    #         if i == j:
    #             res = 0
    #         else:
    #             p = s[:, i] - s[:, j]
    #             res = torch.exp(-torch.sqrt(torch.sum(p * p)) / (2 * var))
    #         A[i, j] = res
    return A

from kmeans_pytorch import kmeans

def finalclustering(s, non0_coord, k, var):
    n = s.shape[1]
    A = affinity(s, non0_coord, var)
    # print(A.shape)

    D = torch.zeros((n, n))
    # obtain D^-1
    for i in range(n):
        D[i, i] = 1 / (A[i].sum())


    L = torch.sqrt(D)@(A)@(torch.sqrt(D))
    # print(L.shape)
    value, vector = torch.linalg.eig(L)
    # print(vector.shape)
    # print(value.shape)
    # print(np.argsort(value))
    idx = np.argsort(value)#torch.argsort(value, dim=0, descending=True)
    # print(idx)
    value = value[idx]
    vector = vector[idx, :]

    X = vector[:, :k]

    Y = X / torch.sqrt(torch.sum(X**2, 0))
    print(Y)
    print(Y.float())
    print()
    # raise "fdsffsdfsd"
    clusters, data = kmeans(
        X=Y.float(),
        num_clusters=k.numpy(),
        # distance='cosine',
        # cluster_centers=[],
        tol=0,
        tqdm_flag=True,
        iter_limit=1000,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.1)
    # clusters, data = kmeans(Y, k, max_iter=20)
    # final = torch.concatenate((s, clusters.reshape((len(clusters), 1))), axis=1)
    return clusters, data, clusters

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([b,g,r])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_output_img(msk, seg, inst_num, coord):

    print("mask shape --> {}".format(msk.shape))
    print("seg shape --> {}".format(seg.shape))
    print('inst_num = {}'.format(inst_num))
    print('coord shape --> {}'.format(coord.shape))

    non0 = torch.nonzero(msk)
    # coord_map = coord[:2]
    coord_map = coord[:, non0[:, 0],non0[:, 1]]
    # coord_map = coord[0][msk > 0]
    # coord_map = coord_map[:,:2]
    instance_img = torch.zeros_like(seg)

    # print('coord_map shape --> {}'.format(coord_map))
    # print('non0 shape --> {}'.format(non0))

    final, data, clusters = finalclustering(coord_map,non0.t(), inst_num, .4)
    print(final.shape)
    print(instance_img[non0[:,0],non0[:,1]].shape)
    instance_img[non0[:,0],non0[:,1]] = final+1
    print(instance_img.shape)
    plt.imshow(instance_img)
    plt.show()

    cl, c = kmeans(
        X=coord,
        num_clusters=inst_num,
        # distance='cosine',
        cluster_centers=[],
        tol=1e-10,
        tqdm_flag=True,
        iter_limit=100,
        device=torch.device('cuda'),
        gamma_for_soft_dtw=0.1)
    plt.imshow(cl.reshape(-1, 64))
    plt.show()
    # plt.imshow(val_ds[i][1]['h_ins'])
    raise "dvxvdxvxdv"


msk = val_ds[0][1]['mask']
seg = val_ds[0][1]['seg']
inst_num = val_ds[0][1]['num_ins']
coord = val_ds[0][1]['h_cor']
get_output_img(msk, seg, inst_num, coord)
