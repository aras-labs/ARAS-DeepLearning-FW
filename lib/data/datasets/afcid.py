import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
from torchvision.io import read_image
from lib.data.utils import download_file_from_google_drive
import zipfile


class AFCID(torch.utils.data.Dataset):
    def __init__(self, root, train='train', transforms=None, download=False):
        self.root = os.path.join(root, 'AFCID-v1', train)
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        if download or not os.path.exists(os.path.join(root, 'AFCID-v1/')):
            try:
                os.makedirs(root)
                print("Directory ", root, " Created ")
            except FileExistsError:
                print("Directory ", root, " already exists")

            # download and extract Dataset files
            print("Downloading ...")
            download_file_from_google_drive('1TrdhPy-X5XPWx6_EQow55hD8OKkdFr5J', os.path.join(root, 'AFCID-v1.zip'))
            print("Extracting ...")
            with zipfile.ZipFile(os.path.join(root, 'AFCID-v1.zip'), 'r') as f:
                f.extractall(path=os.path.join(root, 'AFCID-v1'))

        self.ann_path = os.path.join(self.root, "_annotations.coco.json")
        with open(self.ann_path, 'r') as f:
            self.ann = json.load(f)

        self.data = []
        for img in self.ann['images']:
            image_name = img['file_name']
            boxes = []
            labels = []
            area = []
            iscrowd = []
            for ann in self.ann['annotations']:
                if img['id'] == ann['image_id']:
                    boxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
                    labels.append(ann['category_id'])
                    area.append(ann['area'])
                    iscrowd.append(ann['iscrowd'])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.float32)
            image_id = torch.tensor([img['id']])

            target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

            self.data.append({'image_name': image_name, 'target': target})

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.data[idx]['image_name'])
        # img = read_image(img_path)
        img = Image.open(img_path).convert("RGB")
        target = self.data[idx]['target']

        target["image_id"] = torch.tensor([idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    from core.config import cfg
    import lib.engine.detection.transforms as T
    from lib.utils import show, dataset_visualization
    from lib.system.identification import merge_system_environment_to_cfg


    def get_transform():
        transforms = [T.RandomHorizontalFlip(), T.Color(),
                      T.RandomZoomOut(side_range=(1., 1.5)), T.RandomIoUCrop(), T.ToTensor()]
        return T.Compose(transforms)


    cfg = merge_system_environment_to_cfg(cfg)
    DATASET_PATH = cfg.DATASET.PATH
    train_ds = AFCID(DATASET_PATH, train='train', transforms=get_transform())
    # show(train_ds[0][0])
    dataset_visualization(train_ds[0])

