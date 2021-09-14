from torch.utils.data import DataLoader

from lib.data.datasets import *
from lib.core import *

import lib.utils.optimizer as opt
from lib.utils.environment import init_env

from core.config import cfg
import lib.engine.detection.utils as utils
import lib.engine.detection.transforms as T
from lib.engine.detection.engine import train_one_epoch, evaluate


def get_transform(train=True):
    transforms = []
    if train:
        transforms = [T.RandomHorizontalFlip(),
                      # T.RandomZoomOut(side_range=(1., 1.5)),
                      # T.RandomIoUCrop()
                      ]
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


if __name__ == '__main__':
    device, writer = init_env(cfg, has_writer=True)
    # get the model
    model = eval("{}".format(cfg.MODEL.NAME))(cfg.MODEL.PARAM)
    # move model to the right device
    model.to(device)

    DATASET_PATH = cfg.DATASET.PATH
    aug_flag = (cfg.LOGGING.COMMENT != 'no augment')
    train_ds = eval(cfg.DATASET.NAME)(DATASET_PATH, train='train', transforms=get_transform(aug_flag))
    val_ds = eval(cfg.DATASET.NAME)(DATASET_PATH, train='valid', transforms=get_transform(False))
    test_ds = eval(cfg.DATASET.NAME)(DATASET_PATH, train='test', transforms=get_transform(False))

    NUM_WORKERS = cfg.SYSTEM.NUM_WORKERS
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True,
        collate_fn=utils.collate_fn)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn)
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn)

    print("number of train samples :{}, Number of validation samples:{}"
          .format(len(train_ds), len(val_ds)))
    print("number of train batch :{}, Number of validation batch:{}"
          .format(len(train_dl), len(val_dl)))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_ft = opt.get_optimizer(params, cfg.TRAIN.PARAM.OPTIMIZER, cfg.TRAIN.PARAM.OPTIMIZER_PARAM)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft,
                                                   step_size=3, gamma=0.1)
    num_epochs = cfg.TRAIN.EPOCHS
    best = 0
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer_ft, train_dl, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        val_eval = evaluate(model, val_dl, device=device, ann_path=val_ds.ann_path).coco_eval['bbox'].stats
        test_eval = evaluate(model, test_dl, device=device, ann_path=test_ds.ann_path).coco_eval['bbox'].stats
        if cfg.LOGGING.BEST:
            if val_eval[0] > best:
                best = val_eval[0]
                torch.save(model.state_dict(), os.path.join(cfg.LOGGING.WEIGHT_FOLDER,
                                                            'best.ckp'))
            torch.save(model.state_dict(), os.path.join(cfg.LOGGING.WEIGHT_FOLDER,
                                                        'last.ckp'))
        else:
            torch.save(model.state_dict(), os.path.join(cfg.LOGGING.WEIGHT_FOLDER,
                                                        'epoch_{}.ckp'.format(epoch)))
        writer.add_scalar('val/mAP@0.5:0.95', val_eval[0], epoch, description=cfg.LOGGING.COMMENT)
        writer.add_scalar('val/mAP@0.5', val_eval[1], epoch, description=cfg.LOGGING.COMMENT)
        writer.add_scalar('val/mAP@0.75', val_eval[2], epoch, description=cfg.LOGGING.COMMENT)

        writer.add_scalar('test/mAP@0.5:0.95', test_eval[0], epoch, description=cfg.LOGGING.COMMENT)
        writer.add_scalar('test/mAP@0.5', test_eval[1], epoch, description=cfg.LOGGING.COMMENT)
        writer.add_scalar('test/mAP@0.75', test_eval[2], epoch, description=cfg.LOGGING.COMMENT)
        writer.flush()
