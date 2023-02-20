import copy
import operator
from collections import OrderedDict
from typing import (
    List, Dict, Any, Callable
)

import torch
from torch.fx import (
    GraphModule
)
from torch.quantization import (
    propagate_qconfig_,
    swap_module
)
from torch.nn.intrinsic import (
    _FusedModule
)
from torch.quantization.quantization_mappings import (
    get_default_qat_module_mappings,
    get_default_static_quant_module_mappings
)
from torch.quantization.utils import (
    get_combined_dict
)
from torch.quantization.fx.qconfig_utils import (
    get_flattened_qconfig_dict
)
from torch.quantization.quantize_fx import (
    _fuse_fx
)

from mqbench.utils import getitem2node
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType


@register_model_quantizer(BackendType.Tensorrt)
@register_model_quantizer(BackendType.NNIE)
class ModelQuantizer(object):
    """General model quantizer class.
    First, replace common float module to nn.qat.modules to make weight fake
    quantized.
    Second, insert activation fake quantize node before specific layers. Layer
    type is defined in function_type_to_quant_input / module_type_to_quant_input.
    We only quantize the inputs of layers and leave the output not quantized
    since it is next layer's input.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        self.additional_function_type = extra_quantizer_dict.get('additional_function_type', []) # 额外需要量化的function类型、model类型
        self.additional_module_type = extra_quantizer_dict.get('additional_module_type', ())
        self.additional_fuser_method_mapping = extra_fuse_dict.get('additional_fuser_method_mapping', {})# 额外需要融合的融合方法和融合模式
        self.additional_fusion_pattern = extra_fuse_dict.get('additional_fusion_pattern', {})
        self.additional_qat_module_mapping = extra_fuse_dict.get('additional_qat_module_mapping', {})
        self.additional_node_name = extra_quantizer_dict.get('additional_node_name', [])
        self.exclude_module_name = extra_quantizer_dict.get('exclude_module_name', [])
        self.exclude_function_type = extra_quantizer_dict.get('exclude_function_type', [])
        self.exclude_node_name = extra_quantizer_dict.get('exclude_node_name', [])
        self.extra_fuse_dict = extra_fuse_dict

    def prepare(self, model: GraphModule, qconfig):
        model = _fuse_fx(model, self.extra_fuse_dict) # 准确量化模型前，先进行一些图模型的融合 
        model = self._weight_quant(model, qconfig) # 对权重进行量化, 主要是将module转化为qatModule  
        model = self._insert_fake_quantize_for_act_quant(model, qconfig) # 这里就是对graph进行修改了，然后插入伪量化结点
        return model

    def _insert_fake_quantize_for_act_quant(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer" # 后激活伪量化的后缀
        node_to_quantize_output = self._find_act_quants(model) # 找到所有需要量化的结点，相当于对它们的输出进行量化。
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys() # 

        for node in node_to_quantize_output:
            fake_quantizer = qconfig.activation() # 通过 量化的配置实例qconfig 的activation()方法
            quantizer_name = node.name + quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer) # 让其包含这个 量化类的属性
            logger.info("Insert act quant {}".format(quantizer_name)) # 
            with graph.inserting_after(node): # 在这个节点后插入
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {}) # 插入 call_module 节点(这个节点的输入参数是node,输出是)
                for _node in nodes: # 插入这个节点以后，还需要将原来使用node作为输入的结点都变成以 inserted_node作为输入,即调用如下函数
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        model.recompile() # 检查一下有没有问题啥的
        model.graph.lint()
        return model

    def _fix_succ_recursivly(self, args, target_node, inserted_node):
        # List / Tuple
        if isinstance(args, (list, tuple)):
            _tmp = list(args)
            for _i, _arg in enumerate(args):
                if _arg == target_node:
                    _tmp[_i] = inserted_node
                elif isinstance(_arg, tuple):
                    _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
                elif isinstance(_arg, list):
                    _tmp[_i] = list(self._fix_succ_recursivly(_arg, target_node, inserted_node))
                elif isinstance(_arg, dict):
                    _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
            return tuple(_tmp)
        # Dict
        elif isinstance(args, dict):
            _tmp = {}
            for k, v in args.items():
                if v == target_node:
                    _tmp[k] = inserted_node
                elif not isinstance(v, torch.fx.node.Node):
                    _tmp[k] = self._fix_succ_recursivly(v, target_node, inserted_node)
                else:
                    _tmp[k] = v
            return _tmp
        else:
            raise NotImplementedError('{} can not be handled now.'.format(type(args)))

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")# 对权重进行量化就会把有参数的module，都转换为 qat Module
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict) # 这个传播量化配置在model的级别下进行
        self._qat_swap_modules(model, self.additional_qat_module_mapping) # 将模型置换为qat的可训练模型。 备注置换操作都不需要改变graph
        return model

    @property
    def implicit_merge_patterns(self) -> list:
        # Layers which do not need quantize among them.
        # In reversed order!
        return [
            (operator.add, operator.mul)
        ]

    def _on_merge_chain(self, modules, pattern, pair, p_pos=0, v_pos=0): # 其中，p_pos表示pattern的索引,v_pos表示变量结点的索引
        if v_pos == len(pair): # 如果变量结点的索引等于pair的长度
            return True
        if p_pos == len(pattern):
            return v_pos == len(pair)
        node = pair[v_pos]
        cur_pattern = pattern[p_pos]
        # Means current node is matched.
        if (node.op == "call_module" and type(modules[node.target]) == cur_pattern) or \
                ((node.op == 'call_function' or node.op == 'call_method') and # 判断当前节点的target是否符合当前的模式
                    node.target == cur_pattern):
            # Means comparing pair.
            if len(pattern) > p_pos and len(pair) > v_pos:
                return self._on_merge_chain(modules, pattern, pair, p_pos + 1, v_pos + 1)
            # Means comparing extra node.
            matched = False
            flatten_args = self._flatten_args(node.args)
            for _arg in flatten_args:
                extra_pair = (*pair, _arg)
                if isinstance(_arg, torch.fx.node.Node) and \
                        self._on_merge_chain(modules, pattern, extra_pair, p_pos + 1, v_pos + 1):
                    matched = True
            return matched
        # Current node is not matched, skip to next.
        else:
            return self._on_merge_chain(modules, pattern, pair, p_pos + 1, v_pos)

    def _is_implicit_merge(self, modules, pair):
        for pattern in self.implicit_merge_patterns: # 判断是否在合并链上，如果在，就返回true。 # 当前的合并为(+,*)这个两个符号
            if self._on_merge_chain(modules, pattern, pair):
                return True
        return False

    @property
    def function_type_to_update_data_struct(self) -> list:
        return [
            'update'
        ]

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            # operator.add, # type(operator.add) == type(torch.add)
            # operator.mul, 
            # torch.add,
            # torch.mul,
            # torch.matmul,
            # torch.nn.functional.adaptive_avg_pool2d, # 
            # torch.nn.functional.interpolate,
            # torch.softmax
        ] + self.additional_function_type

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # # Conv
            # torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d, # 因为会做融合，所以不需要ReLU
            # torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,     # 
            # torch.nn.qat.modules.conv.Conv2d,
            # # ConvTranspose
            # torch.nn.ConvTranspose2d,
            # # Linear
            # torch.nn.qat.modules.linear.Linear,
            # torch.nn.Linear,
            # # Pooling
            # torch.nn.modules.pooling.MaxPool2d,
            # torch.nn.modules.pooling.AvgPool2d,
            # torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # # BN
            # torch.nn.BatchNorm2d,
            # # Prelu mostly do not merge.
            # torch.nn.PReLU,
            # # Upsample,
            # torch.nn.Upsample,
            # torch.nn.Softmax,
        ) + self.additional_module_type

    def _flatten_args(self, node):
        """ 将参数扩散，成为一个列表,不同等级的都在一个列表中
        """
        flattned_args = []
        if isinstance(node, dict):
            for v in node.values():
                flattned_args.extend(self._flatten_args(v))
        elif isinstance(node, tuple) or isinstance(node, list):
            for n in node:
                flattned_args.extend(self._flatten_args(n))
        else:
            flattned_args.extend([node])
        return flattned_args

    def _find_act_quants(self, model: GraphModule) -> List: # 找需要量化的节点
        nodes = list(model.graph.nodes) 
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        g2node = getitem2node(model)
        for node in nodes:
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or # 在指定的 exclude列表中则不考虑
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                logger.info("Exclude skip: {}".format(node.name))
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input) or node.name in self.additional_node_name:
                input_node_list = self._flatten_args(node.args) # 找到这个node的args，相当于以来的输入参数
                input_node_list = [_node for _node in input_node_list if isinstance(_node, torch.fx.node.Node)]
                for _node in input_node_list:
                    if self._is_implicit_merge(modules, (node, _node)):
                        logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                        continue
                    if _node in node_need_to_quantize_output:
                        continue
                    if _node in g2node:
                        _node = g2node[_node]
                    node_need_to_quantize_output.append(_node)
        return node_need_to_quantize_output

    def _qat_swap_modules(self, root: GraphModule, additional_qat_module_mapping: Dict[Callable, Callable]):
        all_mappings = get_combined_dict(# 得到模型置换的转化的映射字典
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        root = self._convert(root, all_mappings, inplace=True) # 得到置换后的模型。
        return root

    def _convert(self, module, mapping=None, inplace=False, scope=''):
        if mapping is None:
            mapping = get_default_static_quant_module_mappings()

        if not inplace:
            module = copy.deepcopy(module) # 如果不是Inplace就深拷贝
        reassign = {}
        for name, mod in module.named_children():# 对子进行递归调用，如果已经没有子模型了，就会直接跳过 TODO 存在bug，如果模型只有一层Linear，就不会被量化
            # fused modules are swapped as one unit
            new_scope = "{}.{}".format(scope, name) if scope != '' else name # 更新scope
            if new_scope in self.exclude_module_name: # 跳过一些模块的置换
                logger.info("Skip quant layer: " + new_scope)
                continue
            if not isinstance(mod, _FusedModule):# _FusedModule继承了Sequential(仅仅当做一个特殊的标记而已),如果不是就递归调用_convert。指定新scope
                self._convert(mod, mapping, True, new_scope)
            reassign[name] = swap_module(mod, mapping, {})
        for key, value in reassign.items():
            module._modules[key] = value

        return module
