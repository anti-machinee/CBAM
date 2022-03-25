from base import BaseDataLoader
from .dataset import FaceDataset


class FaceDataloader(BaseDataLoader):
    def __init__(self, config, logger):
        dataset = FaceDataset(config, logger)
        super(FaceDataloader, self).__init__(dataset, config)
