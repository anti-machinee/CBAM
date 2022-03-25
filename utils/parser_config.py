import os
import logging
from datetime import datetime
import importlib

import torch

from utils import get_config, save_config
from logger import setup_logging


class ConfigParser:
    def __init__(self, arguments):
        self.cfg_path = arguments.config

        # save folder
        self.cfg, self.cfg_dict = self.get_config()
        ckpt_path, loss_path, log_path = self.create_save_folder()
        self.cfg.save_model_dir = ckpt_path
        self.cfg.save_loss_dir = loss_path
        self.cfg.log_dir = log_path

        # log level
        self.log_levels = None
        self.set_log_level()
        self.cfg.logger = self.get_logger("trainer", self.cfg.trainer.args.verbosity)
        self.cfg.freq_log = arguments.freq_log

        # device
        self.cfg.device = torch.device("cuda:{}".format(arguments.local_rank) if torch.cuda.is_available() else "cpu")

        # resume
        if arguments.resume_model is not None:
            self.cfg.resume_model = arguments.resume_model
        else:
            self.cfg.resume_model = None
        if arguments.resume_loss is not None:
            self.cfg.resume_loss = arguments.resume_loss
        else:
            self.cfg.resume_loss = None

    def get_config(self):
        config, config_dict = get_config(self.cfg_path)
        return config, config_dict

    def create_save_folder(self):
        version_name = self.cfg.version_name
        run_id = datetime.now().strftime(r"%d%m%Y_%H%M%S")
        ckpt_path = os.path.join(self.cfg.trainer.args.save_dir, version_name, run_id, "model")
        loss_path = os.path.join(self.cfg.trainer.args.save_dir, version_name, run_id, "loss")
        log_path = os.path.join(self.cfg.trainer.args.save_dir, version_name, run_id, "log")
        save_cfg_path = os.path.join(ckpt_path, self.cfg_path.rsplit("/", 1)[-1])
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(loss_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        save_config(self.cfg_dict, save_cfg_path)
        return ckpt_path, loss_path, log_path

    def prepare_device(self, device_index):
        if device_index == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            device = torch.device("cpu")
            self.cfg.logger.info("Using ***** CPU\n")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_index
            device = torch.device("cuda")
            self.cfg.logger.info(f"Using ***** GPU {device_index}\n")
        return device

    def set_log_level(self):
        setup_logging(self.cfg.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @staticmethod
    def init_obj(argument):
        module_name, class_name = argument.rsplit(".", 1)
        somemodule = importlib.import_module(module_name)
        cls_instance = getattr(somemodule, class_name)
        return cls_instance

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
