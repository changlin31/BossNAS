import torch
import torch.nn as nn

try:
    import apex
except:
    print('apex is not installed')
from openselfsup.models import builder
from openselfsup.models.registry import MODELS



@MODELS.register_module
class SiameseSupernetsNATS(nn.Module):
    """Siamese Supernets for NATS search space.

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
        super(SiameseSupernetsNATS, self).__init__()

        self.start_block = start_block
        self.num_block = num_block

        self.online_backbone = builder.build_backbone(backbone)
        self.target_backbone = builder.build_backbone(backbone)
        self.backbone = self.online_backbone
        self.online_necks = nn.ModuleList()
        self.target_necks = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.online_channels_fix = nn.ModuleList()
        self.target_channels_fix = nn.ModuleList()
        self._op_layers_list = self.online_backbone._op_layers_list
        channel_list = self.online_backbone.candidate_Cs
        for i in self._op_layers_list:
            outc = neck['in_channels']
            online_channel_fix = nn.ModuleList()
            for channel in channel_list:
                online_channel_fix.append(nn.Sequential(nn.Conv2d(channel, outc, kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(outc)))
            self.online_channels_fix.append(online_channel_fix)
            target_channel_fix = nn.ModuleList()
            for channel in channel_list:
                target_channel_fix.append(nn.Sequential(nn.Conv2d(channel, outc, kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(outc)))
            self.target_channels_fix.append(target_channel_fix)
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
        for param_ol, param_tgt in zip(self.online_channels_fix.parameters(),
                                       self.target_channels_fix.parameters()):
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
        self.online_channel_fix = self.online_channels_fix[self.start_block]
        self.target_channel_fix = self.target_channels_fix[self.start_block]

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

    def forward_train(self, img, forward_singleop_online, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        pre_op = -1
        if self.start_block > 0:
            forward_bestpath = []
            for path in self.best_paths:
                for op in path:
                    forward_bestpath.append(op)
            pre_op = forward_bestpath[-1]
            for i, best_path in enumerate(self.best_paths):
                img_v1 = self.online_backbone(img_v1,
                                              start_block=i,
                                              forward_op=forward_bestpath)[0]
                img_v2 = self.online_backbone(img_v2,
                                              start_block=i,
                                              forward_op=forward_bestpath)[0]

        channel_fix_op = forward_singleop_online[sum(self._op_layers_list[:self.start_block+1]) - 1]

        proj_online_v1 = self.online_neck(tuple([self.online_channel_fix[channel_fix_op](self.online_backbone(img_v1,
                                                                                  start_block=self.start_block,pre_op=pre_op,
                                                                                  forward_op=forward_singleop_online)[0])]))[0]
        proj_online_v2 = self.online_neck(tuple([self.online_channel_fix[channel_fix_op](self.online_backbone(img_v2,
                                                                                  start_block=self.start_block,pre_op=pre_op,
                                                                                  forward_op=forward_singleop_online)[0])]))[0]

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
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()

        img_v1, idx_unshuffle_v1 = self._batch_shuffle_ddp(img_v1)
        img_v2, idx_unshuffle_v2 = self._batch_shuffle_ddp(img_v2)
        pre_op = -1
        if self.start_block > 0:
            forward_bestpath = []
            for path in self.best_paths:
                for op in path:
                    forward_bestpath.append(op)
            pre_op = forward_bestpath[-1]
            for i, best_path in enumerate(self.best_paths):
                img_v1 = self.target_backbone(img_v1,
                                              start_block=i,
                                              forward_op=forward_bestpath)[0].clone().detach()
                img_v2 = self.target_backbone(img_v2,
                                              start_block=i,
                                              forward_op=forward_bestpath)[0].clone().detach()

        # print("imgv1:"+ str(img_v1.shape))
        self.forward_op_target = self.forward_op_online
        proj_target_v1 = 0
        proj_target_v2 = 0

        with torch.no_grad():
            for forward_singleop_target in self.forward_op_target:
                channel_fix_op = forward_singleop_target[sum(self._op_layers_list[:self.start_block + 1]) - 1]
                temp_v1 = self.target_neck(tuple([self.target_channel_fix[channel_fix_op](self.target_backbone(img_v1,
                                                                start_block=self.start_block,pre_op=pre_op,
                                                                forward_op=forward_singleop_target)[0])]))[
                    0].clone().detach()
                temp_v2 = self.target_neck(tuple([self.target_channel_fix[channel_fix_op](self.target_backbone(img_v2,
                                                                start_block=self.start_block,pre_op=pre_op,
                                                                forward_op=forward_singleop_target)[0])]))[
                    0].clone().detach()
                temp_v1 = nn.functional.normalize(temp_v1, dim=1)
                temp_v1 = self._batch_unshuffle_ddp(temp_v1, idx_unshuffle_v1)

                temp_v2 = nn.functional.normalize(temp_v2, dim=1)
                temp_v2 = self._batch_unshuffle_ddp(temp_v2, idx_unshuffle_v2)

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
        pre_op = -1
        if self.start_block > 0:
            forward_bestpath = []
            for path in self.best_paths:
                for op in path:
                    forward_bestpath.append(op)
            pre_op = forward_bestpath[-1]
            for i, best_path in enumerate(self.best_paths):
                img_v1 = self.target_backbone(img_v1,
                                              start_block=i,
                                              forward_op=forward_bestpath)[0]
                img_v2 = self.target_backbone(img_v2,
                                              start_block=i,
                                              forward_op=forward_bestpath)[0]

        channel_fix_op = forward_singleop[sum(self._op_layers_list[:self.start_block + 1]) - 1]
        self.target_neck(tuple([self.target_channel_fix[channel_fix_op](self.target_backbone(img_v1,
                                              start_block=self.start_block,pre_op=pre_op,
                                              forward_op=forward_singleop,)[0])]))
        self.target_neck(tuple([self.target_channel_fix[channel_fix_op](self.target_backbone(img_v2,
                                              start_block=self.start_block,pre_op=pre_op,
                                              forward_op=forward_singleop,)[0])]))

        self.online_neck(tuple([self.online_channel_fix[channel_fix_op](self.online_backbone(img_v1,
                                              start_block=self.start_block,pre_op=pre_op,
                                              forward_op=forward_singleop,)[0])]))
        self.online_neck(tuple([self.online_channel_fix[channel_fix_op](self.online_backbone(img_v2,
                                              start_block=self.start_block,pre_op=pre_op,
                                              forward_op=forward_singleop,)[0])]))


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
