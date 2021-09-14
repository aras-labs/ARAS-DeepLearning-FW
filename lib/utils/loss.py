import torch
import torch.nn as nn


def ins_loss(pred, target):
    return 10 * nn.SmoothL1Loss()(pred['h_cor'], target['h_cor']) \
           + nn.MSELoss()(pred['num_ins'], target['num_ins'].float())


class NAN_loss(nn.Module):
    def __init__(self, cfg):
        super(NAN_loss, self).__init__()

        self.loss_ss = get_loss(cfg.TRAIN.PARAM.G1.LOSS, cfg.TRAIN.PARAM.G1.LOSS_PARAM)
        self.loss_gp = get_loss(cfg.TRAIN.PARAM.G2.LOSS, cfg.TRAIN.PARAM.G2.LOSS_PARAM)
        self.loss_ins = get_loss(cfg.TRAIN.PARAM.G3.LOSS, cfg.TRAIN.PARAM.G3.LOSS_PARAM)

    def forward(self, pred, target):
        return self.loss_ss(pred['mask'], torch.unsqueeze(target['mask'], dim=1).float())\
               + self.loss_gp(pred['seg'], target['seg'])\
               + self.loss_ins(pred, target)


def get_loss(loss_fn, args):
    try:
        return eval('nn.{}'.format(loss_fn))(**dict(args))
    except:
        try:
            return eval('{}'.format(loss_fn))(args)
        except:
            return eval('{}'.format(loss_fn))
