import torch
import torch.nn as nn

try:
    import apex
except:
    print('apex is not installed')
from openselfsup.models import builder
from openselfsup.models.registry import MODELS


@MODELS.register_module
class SiameseSupernetsMBConv(nn.Module):
    """Siamese Supernets for MBConv search space.

    BossNAS (https://arxiv.org/abs/2103.12424).

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 start_block,
                 num_block,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 use_fp16=False,
                 update_interval=None,
                 **kwargs):
        super(SiameseSupernetsMBConv, self).__init__()

        self.start_block = start_block
        self.num_block = num_block

        self.online_backbone = builder.build_backbone(backbone)
        self.target_backbone = builder.build_backbone(backbone)
        self.backbone = self.online_backbone
        self.online_necks = nn.ModuleList()
        self.target_necks = nn.ModuleList()
        self.heads = nn.ModuleList()
        neck_in_channel_list = [cfg[0] for cfg in self.online_backbone.block_cfgs]
        for in_channel in neck_in_channel_list:
            neck['in_channels'] = in_channel
            self.online_necks.append(builder.build_neck(neck))
            self.target_necks.append(builder.build_neck(neck))
            self.heads.append(builder.build_head(head))

        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for target_neck in self.target_necks:
            for param in target_neck.parameters():
                param.requires_grad = False

        self.init_weights(pretrained=pretrained)
        self.set_current_neck_and_head()

        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.forward_op_online = None
        self.forward_op_target = None
        self.best_paths = []
        self.optimizer = None
        self.use_fp16 = use_fp16
        self.update_interval = update_interval

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        self.online_backbone.init_weights()  # backbone
        for online_neck in self.online_necks:
            online_neck.init_weights(init_linear='kaiming')  # projection

        for param_ol, param_tgt in zip(self.online_backbone.parameters(),
                                       self.target_backbone.parameters()):
            param_tgt.data.copy_(param_ol.data)
        for param_ol, param_tgt in zip(self.online_necks.parameters(),
                                       self.target_necks.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        for head in self.heads:
            head.init_weights()

    def set_current_neck_and_head(self):
        self.online_neck = self.online_necks[self.start_block]
        self.target_neck = self.target_necks[self.start_block]
        self.head = self.heads[self.start_block]
        self.online_net = nn.Sequential(self.online_backbone, self.online_neck)
        self.target_net = nn.Sequential(self.target_backbone, self.target_neck)

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.

        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, forward_singleop_online, idx=0, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        v2_idx = img.shape[1] // 2
        img_v1 = img[:, idx, ...].contiguous()
        img_v2 = img[:, v2_idx + idx, ...].contiguous()
        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                img_v1 = self.online_backbone(img_v1,
                                              start_block=i,
                                              forward_op=best_path,
                                              block_op=True)[0]
                img_v2 = self.online_backbone(img_v2,
                                              start_block=i,
                                              forward_op=best_path,
                                              block_op=True)[0]

        proj_online_v1 = self.online_neck(self.online_backbone(img_v1,
                                                               start_block=self.start_block,
                                                               forward_op=forward_singleop_online))[0]
        proj_online_v2 = self.online_neck(self.online_backbone(img_v2,
                                                               start_block=self.start_block,
                                                               forward_op=forward_singleop_online))[0]

        loss = self.head(proj_online_v1, self.proj_target_v2)['loss'] + \
               self.head(proj_online_v2, self.proj_target_v1)['loss']

        return loss

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        elif mode == 'target':
            return self.forward_target(img, **kwargs)
        elif mode == 'single':
            return self.forward_single(img, **kwargs)
        else:
            raise Exception("No such mode: {}".format(mode))

    @torch.no_grad()
    def forward_target(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())

        img_v_l = []
        idx_unshuffle_v_l = []
        for idx in range(img.shape[1]):
            img_vi = img[:, idx, ...].contiguous()
            img_vi, idx_unshuffle_vi = self._batch_shuffle_ddp(img_vi)
            img_v_l.append(img_vi)
            idx_unshuffle_v_l.append(idx_unshuffle_vi)

        if self.start_block > 0:
            for idx in range(img.shape[1]):
                for i, best_path in enumerate(self.best_paths):
                    img_v_l[idx] = self.target_backbone(img_v_l[idx],
                                                  start_block=i,
                                                  forward_op=best_path,
                                                  block_op=True)[0]

        self.forward_op_target = self.forward_op_online
        proj_target_v1 = 0
        proj_target_v2 = 0
        v2_idx = img.shape[1]//2
        with torch.no_grad():
            for op_idx, forward_singleop_target in enumerate(self.forward_op_target):
                temp_v1 = self.target_neck(self.target_backbone(img_v_l[op_idx],
                                                                start_block=self.start_block,
                                                                forward_op=forward_singleop_target))[
                    0].clone().detach()
                temp_v2 = self.target_neck(self.target_backbone(img_v_l[v2_idx + op_idx],
                                                                start_block=self.start_block,
                                                                forward_op=forward_singleop_target))[
                    0].clone().detach()
                temp_v1 = nn.functional.normalize(temp_v1, dim=1)
                temp_v1 = self._batch_unshuffle_ddp(temp_v1, idx_unshuffle_v_l[op_idx])

                temp_v2 = nn.functional.normalize(temp_v2, dim=1)
                temp_v2 = self._batch_unshuffle_ddp(temp_v2, idx_unshuffle_v_l[v2_idx + op_idx])

                proj_target_v1 += temp_v1
                proj_target_v2 += temp_v2

        self.proj_target_v1 = proj_target_v1 / (len(self.forward_op_target))
        self.proj_target_v2 = proj_target_v2 / (len(self.forward_op_target))

    def forward_single(self, img, forward_singleop, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()

        if self.start_block > 0:
            for i, best_path in enumerate(self.best_paths):
                img_v1 = self.target_backbone(img_v1,
                                              start_block=i,
                                              forward_op=best_path,
                                              block_op=True)[0]
                img_v2 = self.target_backbone(img_v2,
                                              start_block=i,
                                              forward_op=best_path,
                                              block_op=True)[0]

        self.target_neck(self.target_backbone(img_v1,
                                              start_block=self.start_block,
                                              forward_op=forward_singleop,
                                              block_op=True))
        self.target_neck(self.target_backbone(img_v2,
                                              start_block=self.start_block,
                                              forward_op=forward_singleop,
                                              block_op=True))

        self.online_neck(self.online_backbone(img_v1,
                                              start_block=self.start_block,
                                              forward_op=forward_singleop,
                                              block_op=True))
        self.online_neck(self.online_backbone(img_v2,
                                              start_block=self.start_block,
                                              forward_op=forward_singleop,
                                              block_op=True))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
