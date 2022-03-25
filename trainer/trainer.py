from tqdm import tqdm

import torch
import torch.distributed as dist

from base.base_trainer import BaseTrainer
from utils.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None,
                 evaluator=None, sampler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, sampler)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = len(self.data_loader) // config.freq_log

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.rank = dist.get_rank()
        self.evaluator = evaluator

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.criterion.train()
        self.train_metrics.reset()

        for batch_idx, (data, label) in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
            data = data.to(self.device)
            label = label.to(self.device)

            # forward pass
            self.optimizer.zero_grad()
            _, output = self.model(data)
            # backward pass
            loss = self.criterion(output, label)
            loss.backward()
            # update parameters
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, label))

            if batch_idx % self.log_step == 0 and self.rank == 0:
                self.logger.info('Train Epoch: {} {} Loss: {:.9f}  LR: {:.9f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    self.get_lr(self.optimizer)
                ))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation and self.rank == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.criterion.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # for batch_idx, (data, label) in tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader)):
            #     data = data.to(self.device)
            #     label = label.to(self.device)
            #
            #     output = self.model(data)
            #     loss, output_arcface = self.criterion(output, label)
            #
            #     self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            #     self.valid_metrics.update('loss', loss.item())
            #     for met in self.metric_ftns:
            #         self.valid_metrics.update(met.__name__, met(output_arcface, label))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if self.evaluator is not None:
                self.evaluator.process(self.model, self.device, self.logger)

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
