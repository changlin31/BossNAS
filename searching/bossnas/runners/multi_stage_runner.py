# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import platform
import shutil
import time
import warnings

import torch
from mmcv.parallel import is_module_wrapper
from torch.optim import Optimizer

import mmcv
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info, get_time_str
from mmcv.runner import EpochBasedRunner, get_dist_info, LogBuffer


class MultiStageRunner(EpochBasedRunner):
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 num_stages=4,
                 max_epochs=120):
        # ---- BaseRunner init ----
        if batch_processor is not None:
            if not callable(batch_processor):
                raise TypeError('batch_processor must be callable, '
                                f'but got {type(batch_processor)}')
            # warnings.warn('batch_processor is deprecated, please implement '
            #               'train_step() and val_step() in the model instead.')  # suppressed
            # raise an error is `batch_processor` is not None and
            # `model.train_step()` exists.
            if is_module_wrapper(model):
                _model = model.module
            else:
                _model = model
            if hasattr(_model, 'train_step') or hasattr(_model, 'val_step'):
                raise RuntimeError(
                    'batch_processor and model.train_step()/model.val_step() '
                    'cannot be both available.')
        else:
            assert hasattr(model, 'train_step')

        # check the type of `optimizer`
        if isinstance(optimizer, dict):
            for name, optim in optimizer.items():
                if not isinstance(optim, Optimizer):
                    raise TypeError(
                        f'optimizer must be a dict of torch.optim.Optimizers, '
                        f'but optimizer["{name}"] is a {type(optim)}')
        elif not isinstance(optimizer, Optimizer) and optimizer is not None:
            raise TypeError(
                f'optimizer must be a torch.optim.Optimizer object '
                f'or dict or None, but got {type(optimizer)}')

        # check the type of `logger`
        if not isinstance(logger, logging.Logger):
            raise TypeError(f'logger must be a logging.Logger object, '
                            f'but got {type(logger)}')

        # check the type of `meta`
        if meta is not None and not isinstance(meta, dict):
            raise TypeError(
                f'meta must be a dict or None, but got {type(meta)}')

        self.model = model
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.logger = logger
        self.meta = meta

        # create work_dir
        if mmcv.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            mmcv.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0

        if max_epochs is not None and max_iters is not None:
            raise ValueError(
                'Only one of `max_epochs` or `max_iters` can be set.')

        self._max_epochs = max_epochs
        self._max_iters = max_iters
        # TODO: Redesign LogBuffer, it is not flexible and elegant enough
        self.log_buffer = LogBuffer()
        # ---- end BaseRunner init ----

        self.num_stages = num_stages
        self._max_epochs = max_epochs
        assert self._max_epochs % num_stages == 0
        self.epochs_per_stage = self._max_epochs // num_stages
        self.just_resumed = False
        if hasattr(self.model, 'module'):
            self._epoch = self.model.module.start_block * self.epochs_per_stage
        else:
            self._epoch = self.model.start_block * self.epochs_per_stage

    @property
    def epoch(self):
        return self._epoch % self.epochs_per_stage

    @property
    def iter(self):
        return self._iter % self.iters_per_stage

    @property
    def max_epochs(self):
        return self.epochs_per_stage

    @property
    def max_iters(self):
        return self.iters_per_stage

    @property
    def stage(self):
        return int(self._epoch // self.epochs_per_stage + 1)

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.log_buffer.update({'Stage': self.stage})
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        if self.just_resumed:
            self.call_hook('after_train_epoch')
            self._epoch += 1
            self.just_resumed = False
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.iters_per_stage = self._max_iters // self.num_stages
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs
            assert self._max_epochs % self.num_stages == 0
            self.epochs_per_stage = self._max_epochs // self.num_stages

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                self.iters_per_stage = self._max_iters // self.num_stages
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self._epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self._epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='stage{}_epoch{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self._epoch + 1, iter=self._iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self._epoch + 1, iter=self._iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.stage, self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='cpu'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch'] - 1
        self.epochs_per_stage = self._max_epochs // self.num_stages
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')
        if hasattr(self.model, 'module'):
            print('start_block', self.model.module.start_block, 'stage', self.stage)
            self.model.module.start_block = self.stage - 1
        else:
            print('start_block', self.model.start_block, 'stage', self.stage)
            self.model.start_block = self.stage - 1
        self.logger.info('resumed stage %d, epoch %d', self.stage, self.epoch + 1)
        self.just_resumed = True
