from typing import Optional, Type

import torch
import torch.nn as nn
from torch.quantization.fx.fusion_patterns import ConvBNReLUFusion, ModuleReLUFusion
from torch.quantization.fx.quantization_types import QuantizerCls
from torch.fx.graph import Node

import mqbench.nn as qnn
import mqbench.nn.intrinsic as qnni  # 包含一些用于融合的中间类，都是继承 Sequential实现，通过把多个的模块比方说 Conv、BN、ReLU三个行成一个ConvBNReLU2D(这个类本质上就是继承了一个Sequential)
import mqbench.nn.intrinsic.qat as qnniqat # qnniqat包括了一些用来量化感知训练的模块,如ConvBNReLU(和上面的只是名字一样，但是这个可以用来进行量化感知训练)
from mqbench.utils.fusion import fuse_deconv_bn_eval # 自如其名就是对反卷积和BN之间做一个融合，将BN参数融合进反卷积中，就不用使用BN推断了。
from mqbench.nn.modules import FrozenBatchNorm2d #


class ConvFreezebnReLUFusion(ConvBNReLUFusion): # 冻结了BN的  Conv、BN、ReLU的融合方式。 Fusion类是通过传递Nodes和quantizer来完成量化结点的插入的
    def __init__(self, quantizer: QuantizerCls, node: Node):
        super(ConvBNReLUFusion, self).__init__(quantizer, node)
        self.relu_node = None
        self.bn_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and type(quantizer.modules[node.target]) == torch.nn.ReLU):
            self.relu_node = node
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, FrozenBatchNorm2d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]

def fuse_linear_bn(linear, bn):
    r"""Given the linear and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type Linear
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Linear(10, 20)
        >>> b1 = nn.BatchNorm1d(20)
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    assert(linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."

    if linear.training:
        assert bn.affine, 'Only support fusing BatchNorm1d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm1d with tracking_running_stats set to True'
        return qnn.intrinsic.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_deconv_bn(deconv, bn):
    assert(deconv.training == bn.training),\
        'DeConv and BN must be in the same mode (train or eval)'

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_bn_relu(deconv, bn, relu):
    assert(deconv.training == bn.training == relu.training),\
        "DeConv and BN both must be in the same mode (train or eval)."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)



def fuse_conv_freezebn(conv, bn):
    assert(bn.training is False), "Freezebn must be eval."

    fused_module_class_map = {
        nn.Conv2d: qnni.ConvFreezebn2d,
    }

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        return fused_module_class(conv, bn)
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)

def fuse_conv_freezebn_relu(conv, bn, relu):
    assert(conv.training == relu.training and bn.training is False), "Conv and relu both must be in the same mode (train or eval) and bn must be eval."
    fused_module : Optional[Type[nn.Sequential]] = None
    if conv.training:
        map_to_fused_module_train = {
            nn.Conv2d: qnni.ConvFreezebnReLU2d,
        }
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        fused_module = map_to_fused_module_train.get(type(conv), None)
        return fused_module(conv, bn, relu)
    else:
        map_to_fused_module_eval = {
            nn.Conv2d: nn.intrinsic.ConvReLU2d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return fused_module(fused_conv, relu)


def fuse_deconv_freezebn(deconv, bn):
    assert(bn.training is False), "Freezebn must be eval."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeFreezebn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_freezebn_relu(deconv, bn, relu):
    assert(deconv.training == relu.training and bn.training is False), "Conv and relu both must be in the same mode (train or eval) and bn must be eval."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeFreezebnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)


fuse_custom_config_dict = {
    "additional_fuser_method_mapping": {# 融合方式映射,这里给出的都是将多个模型，使用一个函数来转化为一个模型(一般是消除BN和ReLU)
        (torch.nn.Linear, torch.nn.BatchNorm1d): fuse_linear_bn,# 这个方法送入Linear和BatchNorm1D并返回融合后的Linear(注意返回的只是一个Linear)
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d): fuse_deconv_bn, # 类似于这种和BatchNorm融合的返回的都是另一个
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_deconv_bn_relu, # 
        (torch.nn.ConvTranspose2d, torch.nn.ReLU): qnni.ConvTransposeReLU2d,
        (nn.Conv2d, FrozenBatchNorm2d, nn.ReLU): fuse_conv_freezebn_relu,
        (nn.Conv2d, FrozenBatchNorm2d): fuse_conv_freezebn,
        (nn.ConvTranspose2d, FrozenBatchNorm2d, nn.ReLU): fuse_deconv_freezebn_relu,
        (nn.ConvTranspose2d, FrozenBatchNorm2d): fuse_deconv_freezebn,
    },
    "additional_fusion_pattern": {
        (torch.nn.BatchNorm1d, torch.nn.Linear): # 也是将多个模型进行融合。使用一个Fusion类从一个结点开始追踪并且返回量化结点
        ConvBNReLUFusion,
        (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (FrozenBatchNorm2d, torch.nn.Conv2d)):
        ConvFreezebnReLUFusion,
        (FrozenBatchNorm2d, torch.nn.Conv2d):
        ConvFreezebnReLUFusion,
        (torch.nn.ReLU, (FrozenBatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvFreezebnReLUFusion,
        (FrozenBatchNorm2d, torch.nn.ConvTranspose2d):
        ConvFreezebnReLUFusion,
    },
    "additional_qat_module_mappings": { # 从一个分组后的模型(比方说LinearBn1D)是一个分组，将这个分组转化为可以被qat量化后的模型(比方说qnniqat.LinearBn1d)
        nn.ConvTranspose2d: qnn.qat.ConvTranspose2d, 
        qnni.LinearBn1d: qnniqat.LinearBn1d,
        qnni.ConvTransposeBn2d: qnniqat.ConvTransposeBn2d,
        qnni.ConvTransposeReLU2d: qnniqat.ConvTransposeReLU2d,
        qnni.ConvTransposeBnReLU2d: qnniqat.ConvTransposeBnReLU2d,
        qnni.ConvFreezebn2d: qnniqat.ConvFreezebn2d,
        qnni.ConvFreezebnReLU2d: qnniqat.ConvFreezebnReLU2d,
        qnni.ConvTransposeFreezebn2d: qnniqat.ConvTransposeFreezebn2d,
        qnni.ConvTransposeFreezebnReLU2d: qnniqat.ConvTransposeFreezebnReLU2d,
        nn.Embedding: qnn.qat.Embedding,
    },
}


def _sort_fusion_patterns(pats):
    """ 小函数主要实现的是将MNoduleReLUFusion(即仅仅设计ReLU和另一个模型)放到 pattern列表的最后
    """
    keys = []
    for key in pats.keys():
        if pats[key] is ModuleReLUFusion:
            keys.append(key)
    for key in keys:
        pats.move_to_end(key)


# Sinse additional_fuser_method_mapping will not be set because fuser.py:54
# do not pass this dict.
from torch.quantization.fuser_method_mappings import DEFAULT_OP_LIST_TO_FUSER_METHOD # 默认的从操作列表到融合方式的映射
from torch.quantization.fx.pattern_utils import DEFAULT_FUSION_PATTERNS # 默认的融合模式，这个已经买通过一个注册器来实现了
from torch.quantization.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS # 这个QAT-map 从一个普通的模型转化为一个可以被qat量化训练的模型
# 下面的就是在默认的映射基础上 继续做一些扩充
DEFAULT_OP_LIST_TO_FUSER_METHOD.update(
    fuse_custom_config_dict['additional_fuser_method_mapping'])
DEFAULT_FUSION_PATTERNS.update(
    fuse_custom_config_dict['additional_fusion_pattern'])
# Make longer matched pattern prior. 尽可能长的匹配pattern,这也就是上面sort函数的作用
# i.e. Conv + BN + Relu should match ConvBnRelu before BNRelu.
# Any thing registered in class ConvBNReLUFusion should be
# proir than class ModuleReLUFusion.
_sort_fusion_patterns(DEFAULT_FUSION_PATTERNS)
DEFAULT_QAT_MODULE_MAPPINGS.update(
    fuse_custom_config_dict['additional_qat_module_mappings'])
