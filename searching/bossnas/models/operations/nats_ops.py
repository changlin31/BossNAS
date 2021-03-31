from copy import deepcopy
import torch.nn as nn


def get_combination(space, num):
  combs = []
  for i in range(num):
    if i == 0:
      for func in space:
        combs.append( [(func, i)] )
    else:
      new_combs = []
      for string in combs:
        for func in space:
          xstring = string + [(func, i)]
          new_combs.append( xstring )
      combs = new_combs
  return combs


class Structure:

    def __init__(self, genotype):
        assert isinstance(genotype, list) or isinstance(genotype, tuple), 'invalid class of genotype : {:}'.format(
            type(genotype))
        self.node_num = len(genotype) + 1
        self.nodes = []
        self.node_N = []
        for idx, node_info in enumerate(genotype):
            assert isinstance(node_info, list) or isinstance(node_info,
                                                             tuple), 'invalid class of node_info : {:}'.format(
                type(node_info))
            assert len(node_info) >= 1, 'invalid length : {:}'.format(len(node_info))
            for node_in in node_info:
                assert isinstance(node_in, list) or isinstance(node_in, tuple), 'invalid class of in-node : {:}'.format(
                    type(node_in))
                assert len(node_in) == 2 and node_in[1] <= idx, 'invalid in-node : {:}'.format(node_in)
            self.node_N.append(len(node_info))
            self.nodes.append(tuple(deepcopy(node_info)))

    def tolist(self, remove_str):
        # convert this class to the list, if remove_str is 'none', then remove the 'none' operation.
        # note that we re-order the input node in this function
        # return the-genotype-list and success [if unsuccess, it is not a connectivity]
        genotypes = []
        for node_info in self.nodes:
            node_info = list(node_info)
            node_info = sorted(node_info, key=lambda x: (x[1], x[0]))
            node_info = tuple(filter(lambda x: x[0] != remove_str, node_info))
            if len(node_info) == 0: return None, False
            genotypes.append(node_info)
        return genotypes, True

    def node(self, index):
        assert index > 0 and index <= len(self), 'invalid index={:} < {:}'.format(index, len(self))
        return self.nodes[index]

    def tostr(self):
        strings = []
        for node_info in self.nodes:
            string = '|'.join([x[0] + '~{:}'.format(x[1]) for x in node_info])
            string = '|{:}|'.format(string)
            strings.append(string)
        return '+'.join(strings)

    def check_valid(self):
        nodes = {0: True}
        for i, node_info in enumerate(self.nodes):
            sums = []
            for op, xin in node_info:
                if op == 'none' or nodes[xin] is False:
                    x = False
                else:
                    x = True
                sums.append(x)
            nodes[i + 1] = sum(sums) > 0
        return nodes[len(self.nodes)]

    def to_unique_str(self, consider_zero=False):
        # this is used to identify the isomorphic cell, which rerquires the prior knowledge of operation
        # two operations are special, i.e., none and skip_connect
        nodes = {0: '0'}
        for i_node, node_info in enumerate(self.nodes):
            cur_node = []
            for op, xin in node_info:
                if consider_zero is None:
                    x = '(' + nodes[xin] + ')' + '@{:}'.format(op)
                elif consider_zero:
                    if op == 'none' or nodes[xin] == '#':
                        x = '#'  # zero
                    elif op == 'skip_connect':
                        x = nodes[xin]
                    else:
                        x = '(' + nodes[xin] + ')' + '@{:}'.format(op)
                else:
                    if op == 'skip_connect':
                        x = nodes[xin]
                    else:
                        x = '(' + nodes[xin] + ')' + '@{:}'.format(op)
                cur_node.append(x)
            nodes[i_node + 1] = '+'.join(sorted(cur_node))
        return nodes[len(self.nodes)]

    def check_valid_op(self, op_names):
        for node_info in self.nodes:
            for inode_edge in node_info:
                # assert inode_edge[0] in op_names, 'invalid op-name : {:}'.format(inode_edge[0])
                if inode_edge[0] not in op_names: return False
        return True

    def __repr__(self):
        return ('{name}({node_num} nodes with {node_info})'.format(name=self.__class__.__name__, node_info=self.tostr(),
                                                                   **self.__dict__))

    def __len__(self):
        return len(self.nodes) + 1

    def __getitem__(self, index):
        return self.nodes[index]

    @staticmethod
    def str2structure(xstr):
        if isinstance(xstr, Structure): return xstr
        assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
        nodestrs = xstr.split('+')
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            inputs = (xi.split('~') for xi in inputs)
            input_infos = tuple((op, int(IDX)) for (op, IDX) in inputs)
            genotypes.append(input_infos)
        return Structure(genotypes)

    @staticmethod
    def str2fullstructure(xstr, default_name='none'):
        assert isinstance(xstr, str), 'must take string (not {:}) as input'.format(type(xstr))
        nodestrs = xstr.split('+')
        genotypes = []
        for i, node_str in enumerate(nodestrs):
            inputs = list(filter(lambda x: x != '', node_str.split('|')))
            for xinput in inputs: assert len(xinput.split('~')) == 2, 'invalid input length : {:}'.format(xinput)
            inputs = (xi.split('~') for xi in inputs)
            input_infos = list((op, int(IDX)) for (op, IDX) in inputs)
            all_in_nodes = list(x[1] for x in input_infos)
            for j in range(i):
                if j not in all_in_nodes: input_infos.append((default_name, j))
            node_info = sorted(input_infos, key=lambda x: (x[1], x[0]))
            genotypes.append(tuple(node_info))
        return Structure(genotypes)

    @staticmethod
    def gen_all(search_space, num, return_ori):
        assert isinstance(search_space, list) or isinstance(search_space,
                                                            tuple), 'invalid class of search-space : {:}'.format(
            type(search_space))
        assert num >= 2, 'There should be at least two nodes in a neural cell instead of {:}'.format(num)
        all_archs = get_combination(search_space, 1)
        for i, arch in enumerate(all_archs):
            all_archs[i] = [tuple(arch)]

        for inode in range(2, num):
            cur_nodes = get_combination(search_space, inode)
            new_all_archs = []
            for previous_arch in all_archs:
                for cur_node in cur_nodes:
                    new_all_archs.append(previous_arch + [tuple(cur_node)])
            all_archs = new_all_archs
        if return_ori:
            return all_archs
        else:
            return [Structure(x) for x in all_archs]


OPS = {
  'nor_conv_7x7' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (7,7), (stride,stride), (3,3), (1,1), affine, track_running_stats),
  'nor_conv_3x3' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (3,3), (stride,stride), (1,1), (1,1), affine, track_running_stats),
  'nor_conv_1x1' : lambda C_in, C_out, stride, affine, track_running_stats: ReLUConvBN(C_in, C_out, (1,1), (stride,stride), (0,0), (1,1), affine, track_running_stats),
  'skip_connect' : lambda C_in, C_out, stride, affine, track_running_stats: Identity() if stride == 1 and C_in == C_out else FactorizedReduce(C_in, C_out, stride, affine, track_running_stats),
}
candidate_Cs = [8, 16, 24, 32, 40, 48, 56, 64]


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, stride, affine, track_running_stats):
    super(FactorizedReduce, self).__init__()
    self.stride = stride
    self.C_in   = C_in
    self.C_out  = C_out
    self.relu   = nn.ReLU(inplace=False)
    if stride == 1:
      self.conv = SlimmableConv2d(candidate_Cs, candidate_Cs, 1, stride=stride, padding=0, bias=not affine)
    else:
      raise ValueError('Invalid stride : {:}'.format(stride))
    self.bn = SwitchableBatchNorm2d(candidate_Cs)

  def forward(self, x, in_idx, out_idx):
    x = self.relu(x)
    out = self.conv(x, in_idx, out_idx)
    out = self.bn(out, out_idx)
    return out

  def extra_repr(self):
    return 'C_in={C_in}, C_out={C_out}, stride={stride}'.format(**self.__dict__)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x, in_idx, out_idx):
    return x


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine, track_running_stats=True):
    super(ReLUConvBN, self).__init__()
    self.relu = nn.ReLU(inplace=False)
    self.sconv = SlimmableConv2d(candidate_Cs, candidate_Cs, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine)
    self.sbn = SwitchableBatchNorm2d(candidate_Cs)

  def forward(self, x, in_idx, out_idx):
    x = self.relu(x)
    x = self.sconv(x, in_idx, out_idx)
    x = self.sbn(x, out_idx)
    return x


class InferCell(nn.Module):

  def __init__(self, genotype, C_in, C_out, stride, affine=True, track_running_stats=True):
    super(InferCell, self).__init__()

    self.layers  = nn.ModuleList()
    self.node_IN = []
    self.node_IX = []
    self.genotype = deepcopy(genotype)
    for i in range(1, len(genotype)):
      node_info = genotype[i-1]
      cur_index = []
      cur_innod = []
      for (op_name, op_in) in node_info:
        if op_in == 0:
          layer = OPS[op_name](C_in , C_out, stride, affine, track_running_stats)
        else:
          layer = OPS[op_name](C_out, C_out,      1, affine, track_running_stats)
        cur_index.append( len(self.layers) )
        cur_innod.append( op_in )
        self.layers.append( layer )
      self.node_IX.append( cur_index )
      self.node_IN.append( cur_innod )
    self.nodes   = len(genotype)
    self.in_dim  = C_in
    self.out_dim = C_out
    self.factorizedreduce = FactorizedReduce(C_in, C_out, stride, affine, track_running_stats)

  def extra_repr(self):
    string = 'info :: nodes={nodes}, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    laystr = []
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      y = ['I{:}-L{:}'.format(_ii, _il) for _il, _ii in zip(node_layers, node_innods)]
      x = '{:}<-({:})'.format(i+1, ','.join(y))
      laystr.append( x )
    return string + ', [{:}]'.format( ' | '.join(laystr) ) + ', {:}'.format(self.genotype.tostr())

  def forward(self, inputs, in_idx, out_idx):
    nodes = [inputs]
    for i, (node_layers, node_innods) in enumerate(zip(self.node_IX,self.node_IN)):
      node_features = 0
      for _il, _ii in zip(node_layers, node_innods):
        if _il == 3 and in_idx != out_idx:
          node_feature = self.factorizedreduce(nodes[_ii], in_idx=in_idx, out_idx=out_idx)
        else:
          node_feature = self.layers[_il](nodes[_ii], in_idx=in_idx if _il in [0,1,3] else out_idx, out_idx=out_idx)
        node_features += node_feature
      # node_features = sum( self.layers[_il](nodes[_ii], in_idx=in_idx if _ii == 0 else out_idx, out_idx=out_idx) for _il, _ii in zip(node_layers, node_innods) )
      nodes.append( node_features )
    return nodes[-1]


class ResNetBasicblock(nn.Module):

  def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
    super(ResNetBasicblock, self).__init__()
    assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
    self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1, affine, track_running_stats)
    self.conv_b = ReLUConvBN(  planes, planes, 3,      1, 1, 1, affine, track_running_stats)

    if stride == 2:
      self.downsample = nn.Sequential(
                           nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                           SlimmableConv2d(candidate_Cs, candidate_Cs, kernel_size=1, stride=1, padding=0, bias=False)
      )
    else:
      self.downsample = None
    self.downsample_s = ReLUConvBN(inplanes, planes, 1, 1, 0, 1, affine, track_running_stats)
    self.in_dim  = inplanes
    self.out_dim = planes
    self.stride  = stride
    self.num_conv = 2

  def extra_repr(self):
    string = '{name}(inC={in_dim}, outC={out_dim}, stride={stride})'.format(name=self.__class__.__name__, **self.__dict__)
    return string

  def forward(self, inputs, in_idx, out_idx):

    basicblock = self.conv_a(inputs, in_idx, out_idx)
    basicblock = self.conv_b(basicblock, out_idx, out_idx)

    if self.downsample is not None:
        residual = self.downsample[0](inputs)
        residual = self.downsample[1](residual, in_idx, out_idx)
    elif in_idx != out_idx:
        residual = self.downsample_s(inputs, in_idx, out_idx)
    else:
        residual = inputs
    return residual + basicblock


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))
        self.bn = nn.ModuleList(bns)
        self.ignore_model_profiling = True

    def forward(self, input, idx):
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        super(SlimmableConv2d, self).__init__(
            max(in_channels_list), max(out_channels_list),
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=max(groups_list), bias=bias)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]

    def forward(self, input, in_idx, out_index):
        self.in_channels = self.in_channels_list[in_idx]
        self.out_channels = self.out_channels_list[out_index]
        self.groups = self.groups_list[in_idx]
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y
