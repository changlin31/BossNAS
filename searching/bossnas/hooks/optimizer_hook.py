import torch
from mmcv.runner import OptimizerHook
try:
    import apex
except:
    print('apex is not installed')


class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training."""

    def __init__(self, update_interval=1, grad_clip=None, coalesce=True, bucket_size_mb=-1, use_fp16=False):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            for group in runner.optimizer.param_groups:  # skip weight decay of unused or zero grad parameters
                for p in group['params']:
                    if not p.grad is None and torch.sum(torch.abs(p.grad.data)) == 0.0:
                        p.grad = None
            runner.optimizer.step()
            runner.optimizer.zero_grad()
