import torch
from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, config):
        if config.sampler:
            self.sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            self.sampler = None
        self.init_kwargs = {
            "batch_size": config.batch_size,
            "shuffle": config.shuffle,
            "sampler": self.sampler,
            "num_workers": config.num_workers,
            # "collate_fn": dataset.collate_fn,
            "drop_last": config.drop_last
        }
        super(BaseDataLoader, self).__init__(dataset=dataset, **self.init_kwargs)
