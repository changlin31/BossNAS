from mmcv.runner import Hook
from mmcv.parallel import is_module_wrapper

from openselfsup.hooks.registry import HOOKS


@HOOKS.register_module
class RandomPathHook(Hook):
    """Random Path Sampling Hook, used in HytraSupernet."""
    def __init__(self, update_interval=1, **kwargs):
        self.update_interval = update_interval

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                model_ = runner.model.module
            else:
                model_ = runner.model
            model_.forward_op_online = model_.online_backbone.set_forward_cfg(method='random', start_block=model_.start_block)
            model_.forward_op_target = model_.online_backbone.set_forward_cfg(method='random', start_block=model_.start_block)


@HOOKS.register_module
class FairPathHook(Hook):
    """Fair Path Sampling Hook, see FairNAS: https://arxiv.org/abs/1907.01845 """

    def __init__(self, update_interval=1, **kwargs):
        self.update_interval = update_interval

    def before_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                model_ = runner.model.module
            else:
                model_ = runner.model
            model_.forward_op_online = model_.online_backbone.set_forward_cfg(method='fair')
            model_.forward_op_target = model_.online_backbone.set_forward_cfg(method='fair')
