import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

from utils.utils import get_args
from utils.parser_config import ConfigParser


@record
def main():
    # setup cuda
    torch.backends.cudnn.benchmark = True

    # get argument from CLI
    args = get_args()

    # parse config file
    cfg_parser = ConfigParser(args)
    cfg = cfg_parser.cfg

    # create default process group for distributed training
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            # world_size=-1,
                            # rank=rank,
                            )
    torch.cuda.set_device(args.local_rank)

    # initialize instance
    # 1. dataset
    train_loader = cfg_parser.init_obj("dataloader." + cfg.train_loader.type)(cfg.train_loader.args, logger=cfg.logger)
    rank = dist.get_rank()
    # 2. model
    model = cfg_parser.init_obj("models." + cfg.model.type)(**cfg.model.args)
    model.to(cfg.device)
    # 3.loss
    criterion = cfg_parser.init_obj("models." + cfg.loss.type)()
    criterion.to(cfg.device)
    # 4. optimizer
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    criterion_params = filter(lambda p: p.requires_grad, criterion.parameters())
    optimizer = cfg_parser.init_obj("models." + cfg.optimizer.type)(**cfg.optimizer.args,
                                                                    params=[{'params': model_params},
                                                                            {'params': criterion_params}])
    # 5. learning rate scheduler
    lr_scheduler = cfg_parser.init_obj("models." + cfg.lr_scheduler.type)(optimizer=optimizer,
                                                                          **cfg.lr_scheduler.args)
    # 6. metric
    metrics = [cfg_parser.init_obj("models." + cfg.metric.type)]
    # 7. evaluation

    # 8. trainer
    trainer = cfg_parser.init_obj("trainer." + cfg.trainer.type)(model,
                                                                 criterion,
                                                                 metrics,
                                                                 optimizer,
                                                                 cfg,
                                                                 cfg.device,
                                                                 train_loader,
                                                                 valid_data_loader=True,  # still validate
                                                                 lr_scheduler=lr_scheduler,
                                                                 evaluator=None,
                                                                 sampler=train_loader.sampler,
                                                                 )

    # train
    trainer.train()


if __name__ == '__main__':
    main()
