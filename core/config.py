from yacs.config import CfgNode as CN

_C = CN()

_C.LOGGING = CN()
_C.LOGGING.HOST = ''
_C.LOGGING.TIME = ''
_C.LOGGING.WEIGHT_FOLDER = './'
_C.LOGGING.LOG_DIR = './'
_C.LOGGING.COMMENT = ''
_C.LOGGING.BEST = False

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.DEVISE = 'cuda'
_C.SYSTEM.CUDA_VISIBLE_DEVICES = '0,2,3,4,5,6,7,8'
_C.SYSTEM.MAX_BATCH_SIZE = 16

_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'NAN'
# _C.MODEL.G1 = CN()
# _C.MODEL.G1.BACKBONE = 'Unet'
# _C.MODEL.G1.USE_DISCRIMINATOR = False
# _C.MODEL.G1.HEAD = False
# _C.MODEL.G1.HEAD_PARAM = CN(new_allowed=True)
# _C.MODEL.G1.BACKBONE_PARAM = CN(new_allowed=True)
# _C.MODEL.G2 = CN()
# _C.MODEL.G2.BACKBONE = 'Unet'
# _C.MODEL.G2.USE_DISCRIMINATOR = False
# _C.MODEL.G2.HEAD = False
# _C.MODEL.G2.HEAD_PARAM = CN(new_allowed=True)
# _C.MODEL.G2.BACKBONE_PARAM = CN(new_allowed=True)
# _C.MODEL.G3 = CN()
# _C.MODEL.G3.BACKBONE = 'Unet'
# _C.MODEL.G3.USE_DISCRIMINATOR = False
# _C.MODEL.G3.BACKBONE_PARAM = CN(new_allowed=True)
# _C.MODEL.G3.HEAD = 'CoordPlusNum'
# _C.MODEL.G3.HEAD_PARAM = CN(new_allowed=True)

_C.TRAIN = CN()
_C.TRAIN.INPUT_SIZE = (512, 512)
_C.TRAIN.BATCH_SIZE = 10
_C.TRAIN.EPOCHS = 50
_C.TRAIN.STEP = 'e2e'  # options ['S1', 'S2', 'S3', 'e2e']
_C.TRAIN.PARAM = CN(new_allowed=True)
# _C.TRAIN.G1 = CN(new_allowed=True)
# _C.TRAIN.G1.OPTIMIZER = 'Adam'
# _C.TRAIN.G1.OPTIMIZER_PARAM = CN(new_allowed=True)
# _C.TRAIN.G1.LOSS = 'SmoothL1Loss'
# _C.TRAIN.G1.LOSS_PARAM = CN(new_allowed=True)
#
# _C.TRAIN.G2 = CN(new_allowed=True)
# _C.TRAIN.G2.OPTIMIZER = 'Adam'
# _C.TRAIN.G2.OPTIMIZER_PARAM = CN(new_allowed=True)
# _C.TRAIN.G2.LOSS = 'SmoothL1Loss'
# _C.TRAIN.G2.LOSS_PARAM = CN(new_allowed=True)
#
# _C.TRAIN.G3 = CN(new_allowed=True)
# _C.TRAIN.G3.OPTIMIZER = 'Adam'
# _C.TRAIN.G3.OPTIMIZER_PARAM = CN(new_allowed=True)
# _C.TRAIN.G3.LOSS = 'SmoothL1Loss'
# _C.TRAIN.G3.LOSS_PARAM = CN(new_allowed=True)

_C.DATASET = CN()
_C.DATASET.NAME = 'CIHP'
_C.DATASET.PATH = './dataset'

cfg = _C
