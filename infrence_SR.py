import time
from torch.utils.data import DataLoader
from lib.data.datasets import *
from lib.core import *

from lib.utils.environment import init_env

from core.config import cfg
import lib.engine.detection.utils as utils
import lib.engine.detection.transforms as T


def get_transform():
    transforms = [T.ToTensor()]
    return T.Compose(transforms)


if __name__ == '__main__':
    device, writer = init_env(cfg, has_writer=True)
    # get the model
    model = eval("{}".format(cfg.MODEL.NAME))(cfg.MODEL.PARAM)
    # move model to the right device
    model.to(device)

    DATASET_PATH = cfg.DATASET.PATH
    test_ds = eval(cfg.DATASET.NAME)(DATASET_PATH, train='test', transforms=get_transform())

    NUM_WORKERS = cfg.SYSTEM.NUM_WORKERS
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=utils.collate_fn)

    print("number of test samples :{}".format(len(test_ds)))
    print("number of train batch :{}".format(len(test_dl)))

    model.eval()
    sum_time = 0
    t = tqdm(test_dl, desc='calculating inference time..')

    for image, _ in t:
        image = list(img.to(device) for img in image)
        st = time.time()
        outputs = model(image)
        sum_time += (time.time() - st)
    print("mean fps:", 1/(sum_time/len(test_dl)))
    print("mean inference time:", sum_time/len(test_dl))
    with open('inference_time.txt', 'a') as f:
        f.write("Model: {}\n".format(cfg.MODEL.NAME))
        f.write("mean fps: {}\n".format(1/(sum_time/len(test_dl))))
        f.write("mean inference time: {}\n".format(sum_time/len(test_dl)))
        f.write("*****************************************************\n")
        f.write("*****************************************************\n")
        f.write("*****************************************************\n")

