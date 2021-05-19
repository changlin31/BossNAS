import datetime
import os
import re

import apex
import mmcv
import torch
import torch.distributed as dist
import yaml
from mmcv.runner import Hook, obj_from_dict
from openselfsup import datasets
from openselfsup.hooks.registry import HOOKS
from openselfsup.utils import optimizers, print_log
from timm.utils import distribute_bn
from torch.utils.data import Dataset


@HOOKS.register_module
class ValBestPathHook(Hook):
    """Validation hook.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataset,
                 bn_dataset,
                 interval,
                 optimizer_cfg,
                 lr_cfg,
                 dist_mode=True,
                 initial=True,
                 resume_best_path='',
                 epoch_per_stage=None,
                 **eval_kwargs):
        if isinstance(dataset, Dataset) and isinstance(bn_dataset, Dataset):
            self.dataset = dataset
            self.bn_dataset = bn_dataset
        elif isinstance(dataset, dict) and isinstance(bn_dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
            self.bn_dataset = datasets.build_dataset(bn_dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()))
        self.bn_data_loader = datasets.build_dataloader(
            self.bn_dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=True,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()))
        self.bn_data = next(iter(self.bn_data_loader))
        self.bn_data = self.bn_data['img']
        del self.bn_data_loader

        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.optimizer_cfg = optimizer_cfg
        self.lr_cfg = lr_cfg
        self.eval_kwargs = eval_kwargs
        self.epoch_per_stage = epoch_per_stage if epoch_per_stage is not None else interval
        if resume_best_path:
            with open(resume_best_path, 'r') as f:
                self.loaded_best_path = yaml.load(f)
        else:
            self.loaded_best_path = []

    def after_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        block_inteval = self.interval[model.start_block] if isinstance(self.interval, list) else self.interval
        if not self.every_n_epochs(runner, block_inteval):
            return
        if len(self.loaded_best_path)-1 >= model.start_block:  # use loaded best path
            model.best_paths = self.loaded_best_path[:model.start_block+1]
            print_log('loaded best paths: {}'.format(model.best_paths), logger='root')
        else:
            self._run_validate(runner)
            print_log('searched best paths: {}'.format(model.best_paths), logger='root')
        print_log('best paths from all workers:')
        print(model.best_paths)
        torch.cuda.synchronize()
        # save best path
        if runner.rank == 0:
            output_dir = os.path.join(runner.work_dir, 'path_rank')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H')
            filename = os.path.join(output_dir, "bestpath_" + str(model.start_block) + "_" + str(runner.epoch) + "_" + str(time_str) + ".yml")
            with open(filename, 'w', encoding='utf8') as f:
                yaml.dump(model.best_paths, f)
        # initialize for next stage
        block_inteval = self.epoch_per_stage[model.start_block] if isinstance(self.epoch_per_stage, list) else self.epoch_per_stage
        if self.every_n_epochs(runner, block_inteval) and model.start_block < model.num_block - 1:
            model.start_block += 1
            forward_index = model.best_paths[-1][-1]
            if forward_index < 4:
                pos = forward_index // 2
            else:
                pos = forward_index - 2
            model.target_backbone.stage_depths[model.start_block] = pos + 1
            model.online_backbone.stage_depths[model.start_block] = pos + 1
            model.set_current_neck_and_head()
            del model.optimizer
            del runner.optimizer
            new_optimizer = build_optimizer(model, self.optimizer_cfg)
            if model.use_fp16:
                model, new_optimizer = apex.amp.initialize(model, new_optimizer,
                                                       opt_level="O1")
                print_log('**** Initializing mixed precision done. ****')
            runner.optimizer = new_optimizer
            model.optimizer = new_optimizer
            # runner.call_hook('before_run') # reinitialize init_lr

        # if model.start_block == model.num_block:
        #     print("finish all blocks")

    def _run_validate(self, runner):
        runner.model.eval()

        results = self.multi_gpu_test(runner, self.data_loader)
        results = sorted(results.items(), key=lambda x: x[1], reverse=False)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if runner.rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H')
            output_dir = os.path.join(runner.work_dir, 'path_rank')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = os.path.join(output_dir, "path_rank_" + str(model.start_block) + "_" + str(time_str) + ".yml")
            with open(filename, 'w', encoding='utf8') as f:
                yaml.dump(results, f)

        block_inteval = self.interval[model.start_block] if isinstance(self.interval, list) else self.interval
        if self.every_n_epochs(runner, block_inteval):
            best_path = results[0][0]
            best_path = [int(i) for i in list(best_path)]

            if len(model.best_paths) == model.start_block + 1:
                model.best_paths.pop()
            model.best_paths.append(best_path)

        runner.model.train()

    def multi_gpu_test(self, runner, data_loader):
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
        model.eval()
        rank = runner.rank
        # print("start_block", model.start_block)
        path_online_results_v1 = {}
        path_online_results_v2 = {}

        avg_temp1 = []
        avg_temp2 = []
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(data_loader))
        # get all path
        all_path = model.online_backbone.get_all_path(start_block=model.start_block)
        for idx, data in enumerate(data_loader):
            with torch.no_grad():
                # val_data
                img = data['img']
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img = img.to(device)
                assert img.dim() == 5, \
                    "Input must have 5 dims, got: {}".format(img.dim())
                origin_img_v1 = img[:, 0, ...].contiguous()
                origin_img_v2 = img[:, 1, ...].contiguous()

                temp1 = 0
                temp2 = 0
                for forward_singleop in all_path:
                    img_v1 = origin_img_v1
                    img_v2 = origin_img_v2
                    op = ''.join([str(i) for i in forward_singleop])
                    if op not in path_online_results_v1.keys():
                        path_online_results_v1[op] = []
                        path_online_results_v2[op] = []

                    update_bn_stats(self.bn_data, runner, forward_singleop)
                    model.eval()
                    if model.start_block > 0:
                        for i, best_path in enumerate(model.best_paths):
                            img_v1 = model.target_backbone(img_v1,
                                                           start_block=i,
                                                           forward_op=best_path,
                                                           block_op=True)[0]
                            img_v2 = model.target_backbone(img_v2,
                                                           start_block=i,
                                                           forward_op=best_path,
                                                           block_op=True)[0]

                    proj_target_v1 = model.target_neck(model.target_backbone(img_v1,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_singleop,
                                                                             block_op=True
                                                                             ))[
                        0].clone().detach()
                    proj_target_v2 = model.target_neck(model.target_backbone(img_v2,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_singleop,
                                                                             block_op=True, ))[
                        0].clone().detach()
                    temp1 += proj_target_v1
                    temp2 += proj_target_v2


                    proj_online_v1 = model.online_neck(model.online_backbone(img_v1,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_singleop,
                                                                             block_op=True
                                                                             ))[0]
                    proj_online_v2 = model.online_neck(model.online_backbone(img_v2,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_singleop,
                                                                             block_op=True
                                                                             ))[0]
                    path_online_results_v1[op].append(proj_online_v1)
                    path_online_results_v2[op].append(proj_online_v2)

                avg_temp1.append(temp1/len(all_path))
                avg_temp2.append(temp2/len(all_path))

            if rank == 0:
                prog_bar.update()

        torch.cuda.synchronize()
        loss_dict = {}
        for op, online_result1 in path_online_results_v1.items():
            online_result2 = path_online_results_v2[op]
            loss_dict[op] = 0
            for i, proj_onlinev1 in enumerate(online_result1):
                proj_onlinev2 = online_result2[i]

                proj_targetv11 = avg_temp1[i]
                proj_targetv22 = avg_temp2[i]

                loss = model.head(proj_onlinev1, proj_targetv22)['loss'] + \
                       model.head(proj_onlinev2, proj_targetv11)['loss']

                loss_dict[op] += reduce_tensor(loss.data, dist.get_world_size()).item()
            loss_dict[op] /= len(online_result1)

        return loss_dict


@HOOKS.register_module
class ValNATSPathHook(Hook):
    """Validation hook.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataset,
                 bn_dataset,
                 interval,
                 optimizer_cfg,
                 lr_cfg,
                 dist_mode=True,
                 initial=True,
                 resume_best_path='',
                 epoch_per_stage=None,
                 **eval_kwargs):
        if isinstance(dataset, Dataset) and isinstance(bn_dataset, Dataset):
            self.dataset = dataset
            self.bn_dataset = bn_dataset
        elif isinstance(dataset, dict) and isinstance(bn_dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
            self.bn_dataset = datasets.build_dataset(bn_dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()))
        self.bn_data_loader = datasets.build_dataloader(
            self.bn_dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=True,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()))
        self.bn_data = next(iter(self.bn_data_loader))
        self.bn_data = self.bn_data['img']
        del self.bn_data_loader

        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.optimizer_cfg = optimizer_cfg
        self.lr_cfg = lr_cfg
        self.eval_kwargs = eval_kwargs
        self.epoch_per_stage = epoch_per_stage if epoch_per_stage is not None else interval
        if resume_best_path:
            with open(resume_best_path, 'r') as f:
                self.loaded_best_path = yaml.load(f)
        else:
            self.loaded_best_path = []

    def after_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        block_inteval = self.interval[model.start_block] if isinstance(self.interval, list) else self.interval
        if not self.every_n_epochs(runner, block_inteval):
            return
        if len(self.loaded_best_path)-1 >= model.start_block:  # use loaded best path
            model.best_paths = self.loaded_best_path[:model.start_block+1]
            print_log('loaded best paths: {}'.format(model.best_paths), logger='root')
        else:
            self._run_validate(runner)
            print_log('searched best paths: {}'.format(model.best_paths), logger='root')
        print_log('best paths from all workers:')
        print(model.best_paths)
        torch.cuda.synchronize()
        # save best path
        if runner.rank == 0:
            output_dir = os.path.join(runner.work_dir, 'path_rank')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H-%M-%S')
            filename = os.path.join(output_dir, "bestpath_" + str(model.start_block) + "_" + str(
                time_str) + ".yml")
            with open(filename, 'w', encoding='utf8') as f:
                yaml.dump(model.best_paths, f)
        # initialize for next stage
        block_inteval = self.epoch_per_stage[model.start_block] if isinstance(self.epoch_per_stage, list) else self.epoch_per_stage
        if self.every_n_epochs(runner, block_inteval) and model.start_block < model.num_block - 1:
            model.start_block += 1
            model.set_current_neck_and_head()
            del model.optimizer
            del runner.optimizer
            new_optimizer = build_optimizer(model, self.optimizer_cfg)
            if model.use_fp16:
                model, new_optimizer = apex.amp.initialize(model, new_optimizer,
                                                       opt_level="O1")
                print_log('**** Initializing mixed precision done. ****')
            runner.optimizer = new_optimizer
            model.optimizer = new_optimizer
            # runner.call_hook('before_run') # reinitialize init_lr

        # if model.start_block == model.num_block:
        #     print("finish all blocks")

    def _run_validate(self, runner):
        runner.model.eval()
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if len(model.best_paths) == model.start_block + 1:
            model.best_paths.pop()
        results = self.multi_gpu_test(runner, self.data_loader)
        results = sorted(results.items(), key=lambda x: x[1], reverse=False)
        if runner.rank == 0:
            time_str = datetime.datetime.strftime(datetime.datetime.now(), '%m-%d-%H-%M-%S')
            output_dir = os.path.join(runner.work_dir, 'path_rank')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = os.path.join(output_dir, "path_rank_" + str(model.start_block) + "_" + str(time_str) + ".yml")
            with open(filename, 'w', encoding='utf8') as f:
                yaml.dump(results, f)

        block_inteval = self.interval[model.start_block] if isinstance(self.interval, list) else self.interval
        if self.every_n_epochs(runner, block_inteval):
            best_path = results[0][0]
            best_path = [int(i) for i in list(best_path)]

            if len(model.best_paths) == model.start_block + 1:
                model.best_paths.pop()
            model.best_paths.append(best_path)

        runner.model.train()

    def multi_gpu_test(self, runner, data_loader):
        if hasattr(runner.model, 'module'):
            model = runner.model.module
        else:
            model = runner.model
        model.eval()
        rank = runner.rank
        # print("start_block", model.start_block)
        path_online_results_v1 = {}
        path_online_results_v2 = {}

        avg_temp1 = []
        avg_temp2 = []
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(data_loader))
        # get all path
        all_path = model.online_backbone.get_all_path(start_block=model.start_block)
        for idx, data in enumerate(data_loader):
            with torch.no_grad():
                # val_data
                img = data['img']
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img = img.to(device)
                assert img.dim() == 5, \
                    "Input must have 5 dims, got: {}".format(img.dim())
                origin_img_v1 = img[:, 0, ...].contiguous()
                origin_img_v2 = img[:, 1, ...].contiguous()
                temp1 = 0
                temp2 = 0
                for forward_singleop in all_path:
                    img_v1 = origin_img_v1
                    img_v2 = origin_img_v2
                    op = ''.join([str(i) for i in forward_singleop])
                    if op not in path_online_results_v1.keys():
                        path_online_results_v1[op] = []
                        path_online_results_v2[op] = []
                    forward_bestpath = []
                    for path in model.best_paths:
                        for _op in path:
                            forward_bestpath.append(_op)
                    for _op in forward_singleop:
                        forward_bestpath.append(_op)
                    update_bn_stats(self.bn_data, runner, forward_bestpath)
                    model.eval()
                    if model.start_block > 0:
                        forward_bestpath_ = []
                        for path in model.best_paths:
                            for _op in path:
                                forward_bestpath_.append(_op)
                        for i, best_path in enumerate(model.best_paths):
                            img_v1 = model.target_backbone(img_v1,
                                                           start_block=i,
                                                           forward_op=forward_bestpath_)[0]
                            img_v2 = model.target_backbone(img_v2,
                                                           start_block=i,
                                                           forward_op=forward_bestpath_)[0]

                    channel_fix_op = forward_bestpath[-1]
                    proj_target_v1 = model.target_neck(tuple([model.target_channel_fix[channel_fix_op]
                                                              (model.target_backbone(img_v1,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_bestpath)[0])]))[
                        0].clone().detach()
                    proj_target_v2 = model.target_neck(tuple([model.target_channel_fix[channel_fix_op](model.target_backbone(img_v2,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_bestpath)[0])]))[
                        0].clone().detach()
                    temp1 += proj_target_v1
                    temp2 += proj_target_v2


                    proj_online_v1 = model.online_neck(tuple([model.online_channel_fix[channel_fix_op](model.online_backbone(img_v1,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_bestpath)[0])]))[0]
                    proj_online_v2 = model.online_neck(tuple([model.online_channel_fix[channel_fix_op](model.online_backbone(img_v2,
                                                                             start_block=model.start_block,
                                                                             forward_op=forward_bestpath)[0])]))[0]
                    path_online_results_v1[op].append(proj_online_v1)
                    path_online_results_v2[op].append(proj_online_v2)

                avg_temp1.append(temp1/len(all_path))
                avg_temp2.append(temp2/len(all_path))

            if rank == 0:
                prog_bar.update()

        torch.cuda.synchronize()
        loss_dict = {}
        for op, online_result1 in path_online_results_v1.items():
            online_result2 = path_online_results_v2[op]
            loss_dict[op] = 0
            for i, proj_onlinev1 in enumerate(online_result1):
                proj_onlinev2 = online_result2[i]

                proj_targetv11 = avg_temp1[i]
                proj_targetv22 = avg_temp2[i]

                loss = model.head(proj_onlinev1, proj_targetv22)['loss'] + \
                       model.head(proj_onlinev2, proj_targetv11)['loss']

                loss_dict[op] += reduce_tensor(loss.data, dist.get_world_size()).item()
            loss_dict[op] /= len(online_result1)

        return loss_dict


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def build_optimizer(model, optimizer_cfg):  # copied from train.py to avoid circular import
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> paramwise_options = {
        >>>     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
        >>>     '\Ahead.': dict(lr_mult=10, momentum=0)}
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001,
        >>>                      paramwise_options=paramwise_options)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, optimizers,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'): # is a multiplier
                            key = key[:-5]
                            assert key in optimizer_cfg, \
                                "{} not in optimizer_cfg".format(key)
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print_log('paramwise_options -- {}: {}={}'.format(
                                name, key, value))

            # otherwise use the global settings
            params.append(param_group)

        optimizer_cls = getattr(optimizers, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


BN_MODULE_TYPES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
)


def update_bn_stats(bn_data, runner, forward_singleop):

    bn_data = bn_data.cuda()
    assert bn_data.dim() == 5, \
        "Input must have 5 dims, got: {}".format(bn_data.dim())
    for layer in runner.model.modules():
        if isinstance(layer, BN_MODULE_TYPES):
            layer.reset_running_stats()
            layer.momentum = 1.
            layer.train()

    with torch.no_grad():  # No need to backward
        runner.model(img=bn_data, mode='single', forward_singleop=forward_singleop)
    torch.cuda.synchronize()
    distribute_bn(runner.model, dist.get_world_size(), reduce=True)
