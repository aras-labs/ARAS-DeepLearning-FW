import matplotlib.pyplot as plt

import numpy as np
import torch
from torchmetrics import IoU
import torch.nn.functional as F
from lib.data.transforms import transforms
from lib.engine.kmeans_pytorch import kmeans


def calc_mean_iou(probas, targs):
    # probas = F.softmax(probas, dim=1)
    iou = IoU(num_classes=20, threshold=0.5, absent_score=1.0, ignore_index=3)
    return iou(probas, targs)


def calc_ap_p(preds, targets, nm_class=20, iou_tresh=0.5):
    nt = targets['num_ins']
    nd = preds['num_ins']
    print(preds['h_cor'].shape)
    x = preds['h_cor'].to(torch.device('cuda:0')).permute(0, 2, 3, 1).reshape(-1, 4)
    #
    cl, c = kmeans(
        X=x,
        num_clusters=int(nd+1),
        # distance='cosine',
        cluster_centers=[],
        tol=0,
        tqdm_flag=False,
        iter_limit=100,
        device=torch.device('cuda'),
        gamma_for_soft_dtw=0.1)
    #
    plt.imshow(cl.reshape(-1, 512))
    plt.show()

    tp = np.zeros(nd.numpy())
    fp = np.zeros(nd.numpy())
    fig, ax = plt.subplots(1, 9)
    t_h_ins = targets['h_ins']
    p_h_ins = preds['h_ins']
    print(p_h_ins.shape)
    p_h_ins = cl.reshape(1, -1, 512)
    det = [False] * nt
    # print(torch.squeeze(p_h_ins, dim=0).shape)
    # ax[0].imshow(torch.squeeze(p_h_ins, 0))

    for i in range(nd):
        p_ins = preds['ins'].clone()
        p_ins[p_h_ins != i + 1] = 0
        p_ins = p_ins % nm_class
        max_iou = -1000

        for j in range(nt):
            t_ins = preds['ins'].clone()
            t_ins[t_h_ins != j + 1] = 0
            t_ins = t_ins % nm_class
            iou = calc_mean_iou(p_ins, t_ins)
            # ax[i*4+j+1].imshow(torch.squeeze(t_ins, 0))
            print(j, p_ins.shape, t_ins.shape, iou)
            if iou > max_iou:
                max_iou = iou
                j_max = j
        if max_iou > iou_tresh and not det[j_max]:
            tp[i] = 1
            det[j_max] = True
        else:
            fp[i] = 1
        # plt.pause(10)
    return tp, fp


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def evaluate(model, data_loader, metrics=None, device='cpu', data_prepare_func=None):
    if metrics is None:
        metrics = []
    model.to(device)
    model.eval()

    running_metrics = {}
    for m in metrics:
        running_metrics[m] = 0

    with torch.set_grad_enabled(False):
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if data_prepare_func is not None:
                input_batch, target_batch = data_prepare_func(input_batch, target_batch)

            if isinstance(input_batch, dict):
                for k in input_batch.keys():
                    input_batch[k] = input_batch[k].to(device)
            else:
                input_batch = input_batch.to(device)

            if isinstance(target_batch, dict):
                for k in target_batch.keys():
                    target_batch[k] = target_batch[k].to(device)
            else:
                target_batch = target_batch.to(device)

            preds = model(input_batch)
            for m in metrics:
                mm = metrics[m](preds, target_batch)
                running_metrics[m] += mm.item()

        for m in metrics:
            running_metrics[m] /= len(data_loader)

    return running_metrics


if __name__ == '__main__':
    from lib.data.datasets import cihp
    from lib.system.identification import merge_system_environment_to_cfg, set_gpu_environment
    from core.config import cfg
    from torch.utils.data import DataLoader
    from tqdm import tqdm, trange

    cfg.merge_from_file('./config/train_parameter.yml')
    cfg = merge_system_environment_to_cfg(cfg)
    cfg.freeze()

    transform = transforms.ToTensor()

    DATASET_PATH = cfg.DATASET.PATH
    train_ds = cihp.CIHP(DATASET_PATH, train=True, transform=transform)
    val_ds = cihp.CIHP(DATASET_PATH, train=False, transform=transform)

    train_dl = DataLoader(train_ds, 1, shuffle=False, num_workers=1, drop_last=True)
    val_dl = DataLoader(val_ds, 1, shuffle=False, num_workers=1, drop_last=True)
    for _, target in val_dl:
        tp_seg, fp_seg = calc_ap_p(target, target)
        print(tp_seg, fp_seg)
        # compute precision recall
        fp_seg = np.cumsum(fp_seg)
        tp_seg = np.cumsum(tp_seg)
        rec_seg = tp_seg / float(target['num_ins'])
        prec_seg = tp_seg / (tp_seg + fp_seg)

        ap_seg = voc_ap(rec_seg, prec_seg)
        print(ap_seg)
        # break
    # data_root = 'D:/Master_Degree/final_project/datasets/LV-MHP-v2/'
    # # set_ in ['train', 'val', 'test_all', 'test_inter_top20', 'test_inter_top10'])
    # set_ = 'val'
    # dat_list = mhp_data.get_data(data_root, set_)
    # # dat_list = pickle.load(open('cache/dat_list_val.pkl'))
    #
    # NUM_CLASSES = 59
    # results_all = get_prediction_from_gt(dat_list, NUM_CLASSES, cache_pkl=False, Sparse=False)
    # eval_seg_ap(results_all, dat_list, nb_class=NUM_CLASSES, ovthresh_seg=0.5, From_pkl=False, Sparse=False)
