import os
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.distributed as dist

from base.base_dataset import BaseDataset
from .augment import Transform


class FaceDataset(BaseDataset):
    def __init__(self, config, logger):
        super(FaceDataset, self).__init__(config, logger)
        self.rank = dist.get_rank()
        if self.rank == 0:
            self.logger.info(f"***** {self.mode.upper()}")
        self.image_paths = self.load_path()
        self.transform = Transform(mode=self.mode)

    def load_path(self):
        image_paths = []
        for label_fol in self.label_folders:
            label_fol = os.path.join(self.root_path, label_fol)
            list_id = os.listdir(label_fol)
            list_id = [int(person_id) for person_id in list_id]
            list_id.sort()
            fol_paths = []
            for person_id in tqdm(list_id, "Loading person_id folder"):
                person_paths = glob(os.path.join(label_fol, str(person_id), "*"))
                fol_paths.extend(person_paths)
            image_paths.extend(fol_paths)
        self.logger.info(f"Total images ***** {len(image_paths)}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image0 = cv2.imread(image_path)
        image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = self.transform(image)
        person_id = int(image_path.split("/")[-2])
        return image, person_id
