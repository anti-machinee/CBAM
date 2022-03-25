import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config, logger):
        super(BaseDataset, self).__init__()
        self.cfg = config
        self.logger = logger

        self.mode = self.cfg.mode
        self.root_path = self.cfg.root_path
        self.label_folders = self.cfg.label_folders
        self.extensions = self.cfg.extensions
        self.image_size = self.cfg.image_size

    def load_path(self):
        raise NotImplemented

    def get_data(self):
        pass

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, index):
        raise NotImplemented
