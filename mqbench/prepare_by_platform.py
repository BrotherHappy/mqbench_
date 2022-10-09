from copy import deepcopy
from enum import Enum
from typing import Any, Dict
import types
import inspect

# TODO  可以增加根据Module类别匹配 Qconfig的格式。
import torch
from torch.fx import Tracer
from torch.fx.graph_module import GraphModule
from torch.quantization.quantize_fx import _swap_ff_with_fxff
from torch.quantization import QConfig


from mqbench.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
)
from mqbench.observer import (
    ClipStdObserver,
    LSQObserver,
    MinMaxFloorObserver,
    MinMaxObserver,
    EMAMinMaxObserver,
    PoTModeObserver,
    EMAQuantileObserver,
    MSEObserver,
    EMAMSEObserver,
)
from mqbench.fuser_method_mappings import fuse_custom_config_dict
from mqbench.utils.logger import logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
from mqbench.scheme import QuantizeScheme

__all__ = ["prepare_by_platform"]


class BackendType(Enum):
    Academic = "Academic"
    Tensorrt = "Tensorrt"
    SNPE = "SNPE"
    PPLW8A16 = "PPLW8A16"
    NNIE = "NNIE"
    Vitis = "Vitis"
    ONNX_QNN = "ONNX_QNN"
    PPLCUDA = "PPLCUDA"
    OPENVINO = "OPENVINO"
    Tengine_u8 = "Tengine_u8"
    Tensorrt_NLP = "Tensorrt_NLP"
    Academic_NLP = "Academic_NLP"


ParamsTable = {
    BackendType.Academic: dict(qtype="affine"),  # noqa: E241
    BackendType.NNIE: dict(
        qtype="nnie",  # noqa: E241
        # NNIE actually do not need w/a qscheme. We add for initialize observer only.
        w_qscheme=QuantizeScheme(
            symmetry=True, per_channel=False, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=True, per_channel=False, pot_scale=False, bit=8
        ),
        default_weight_quantize=NNIEFakeQuantize,
        default_act_quantize=NNIEFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=EMAMinMaxObserver,
    ),
    BackendType.Tensorrt: dict(
        qtype="affine",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=True,
            per_channel=True,
            pot_scale=False,
            bit=8,
            symmetric_range=True,
        ),
        a_qscheme=QuantizeScheme(
            symmetry=True,
            per_channel=False,
            pot_scale=False,
            bit=8,
            symmetric_range=True,
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=EMAMinMaxObserver,
    ),
    BackendType.OPENVINO: dict(
        qtype="affine",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=True, per_channel=True, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=True, per_channel=False, pot_scale=False, bit=8
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=EMAMinMaxObserver,
    ),
    BackendType.SNPE: dict(
        qtype="affine",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=EMAMinMaxObserver,
    ),
    BackendType.PPLW8A16: dict(
        qtype="affine",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=16
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=EMAMinMaxObserver,
    ),
    BackendType.Vitis: dict(
        qtype="vitis",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=True, per_channel=False, pot_scale=True, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=True, per_channel=False, pot_scale=True, bit=8
        ),
        default_weight_quantize=TqtFakeQuantize,
        default_act_quantize=TqtFakeQuantize,
        default_weight_observer=MinMaxFloorObserver,
        default_act_observer=PoTModeObserver,
    ),
    BackendType.ONNX_QNN: dict(
        qtype="affine",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=MinMaxObserver,
    ),
    BackendType.PPLCUDA: dict(
        qtype="affine",  # noqa: E241
        w_qscheme=QuantizeScheme(
            symmetry=False, per_channel=True, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=MinMaxObserver,
    ),
    BackendType.Tengine_u8: dict(
        qtype="affine",
        w_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        a_qscheme=QuantizeScheme(
            symmetry=False, per_channel=False, pot_scale=False, bit=8
        ),
        default_weight_quantize=LearnableFakeQuantize,
        default_act_quantize=LearnableFakeQuantize,
        default_weight_observer=MinMaxObserver,
        default_act_observer=EMAMinMaxObserver,
    ),
}
ParamsTable[BackendType.Tensorrt_NLP] = ParamsTable[BackendType.Tensorrt]
ParamsTable[BackendType.Academic_NLP] = ParamsTable[BackendType.Academic]

ObserverDict = {
    "MinMaxObserver": MinMaxObserver,  # noqa: E241
    "EMAMinMaxObserver": EMAMinMaxObserver,  # More general choice.   # noqa: E241
    "MinMaxFloorObserver": MinMaxFloorObserver,  # For Vitis HW           # noqa: E241
    "PoTModeObserver": PoTModeObserver,  # For Vitis HW           # noqa: E241
    "EMAQuantileObserver": EMAQuantileObserver,  # Quantile observer.     # noqa: E241
    "ClipStdObserver": ClipStdObserver,  # Usually used for DSQ.  # noqa: E241
    "LSQObserver": LSQObserver,  # Usually used for LSQ.  # noqa: E241
    "MSEObserver": MSEObserver,  # noqa: E241
    "EMAMSEObserver": EMAMSEObserver,  # noqa: E241
}

FakeQuantizeDict = {
    "FixedFakeQuantize": FixedFakeQuantize,  # Unlearnable scale/zeropoint  # noqa: E241
    "LearnableFakeQuantize": LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    "NNIEFakeQuantize": NNIEFakeQuantize,  # Quantize function for NNIE   # noqa: E241
    "DoReFaFakeQuantize": DoReFaFakeQuantize,  # Dorefa                       # noqa: E241
    "DSQFakeQuantize": DSQFakeQuantize,  # DSQ                          # noqa: E241
    "PACTFakeQuantize": PACTFakeQuantize,  # PACT                         # noqa: E241
    "TqtFakeQuantize": TqtFakeQuantize,  # TQT                          # noqa: E241
    "AdaRoundFakeQuantize": AdaRoundFakeQuantize,  # AdaRound                     # noqa: E241
    "QDropFakeQuantize": QDropFakeQuantize,  # BRECQ & QDrop                # noqa: E241
}


def get_qconfig_by_platform(deploy_backend: BackendType, extra_qparams: Dict):
    """

    Args:
        deploy_backend (BackendType):
        extra_qparams (dict):

    >>> extra params format: {
            'w_observer': str, weight observer name,
            'a_observer': str, activation observer name,
            'w_fakequantize': str, weight fake quantize function name,
            'w_fakeq_params": dict, params for weight quantize function,
            'a_fakequantize': str, activation fake quantize function name,
            'a_fakeq_params': dict, params for act quantize function,
            if deploy_backend == BackendType.Academic keys below will be used:
            'w_qscheme': {
                'bit': bitwidth,
                'symmetry': whether quantize scheme is symmetric,
                'per_channel': whether quantize scheme is perchannel,
                'pot_scale': whether scale is power of two.
            }
            'a_qscheme': {
                same with w_qscheme.
            }
        }
    """
    w_observer = extra_qparams.get("w_observer", None)
    if w_observer:
        assert w_observer in ObserverDict, "Do not support observer name: {}".format(
            w_observer
        )
        w_observer = ObserverDict[w_observer]
    a_observer = extra_qparams.get("a_observer", None)
    if a_observer:
        assert a_observer in ObserverDict, "Do not support observer name: {}".format(
            a_observer
        )
        a_observer = ObserverDict[a_observer]
    w_fakequantize = extra_qparams.get("w_fakequantize", None)
    if w_fakequantize:
        assert (
            w_fakequantize in FakeQuantizeDict
        ), "Do not support fakequantize name: {}".format(w_fakequantize)
        w_fakequantize = FakeQuantizeDict[w_fakequantize]
    a_fakequantize = extra_qparams.get("a_fakequantize", None)
    if a_fakequantize:
        assert (
            a_fakequantize in FakeQuantizeDict
        ), "Do not support fakequantize name: {}".format(a_fakequantize)
        a_fakequantize = FakeQuantizeDict[a_fakequantize]
    backend_params = ParamsTable[deploy_backend]

    # NNIE backend must use NNIEFakeQuantize but leave observer adjustable.
    if backend_params["qtype"] == "nnie":
        if not w_observer:
            w_observer = backend_params["default_weight_observer"]
        if not a_observer:
            a_observer = backend_params["default_act_observer"]
        w_qscheme = backend_params["w_qscheme"]
        a_qscheme = backend_params["a_qscheme"]
        w_config = backend_params["default_weight_quantize"].with_args(
            observer=w_observer, **w_qscheme.to_observer_params()
        )
        a_config = backend_params["default_act_quantize"].with_args(
            observer=a_observer, **a_qscheme.to_observer_params()
        )
        return QConfig(activation=a_config, weight=w_config)

    # Academic setting should specific quant scheme in config.
    if deploy_backend in [BackendType.Academic, BackendType.Academic_NLP]:
        w_qscheme = QuantizeScheme(**extra_qparams["w_qscheme"])
        a_qscheme = QuantizeScheme(**extra_qparams["a_qscheme"])
    else:
        w_qscheme = extra_qparams.get("w_qscheme", None)
        if w_qscheme is None:
            w_qscheme = backend_params["w_qscheme"]
        else:
            logger.info("Weight Quant Scheme is overrided!")
            w_qscheme = QuantizeScheme(**w_qscheme)
        a_qscheme = extra_qparams.get("a_qscheme", None)
        if a_qscheme is None:
            a_qscheme = backend_params["a_qscheme"]
        else:
            logger.info("Activation Quant Scheme is overrided!")
            a_qscheme = QuantizeScheme(**a_qscheme)

    # Set extra args for observers.
    w_observer_extra_args = extra_qparams.get("w_observer_extra_args", {})
    a_observer_extra_args = extra_qparams.get("a_observer_extra_args", {})
    w_qscheme.kwargs.update(w_observer_extra_args)
    a_qscheme.kwargs.update(a_observer_extra_args)
    # Get weight / act fake quantize function and params. And bias fake quantizer if needed(Vitis)
    if not w_fakequantize:
        w_fakequantize = backend_params["default_weight_quantize"]
    w_fakeq_params = extra_qparams.get("w_fakeq_params", {})
    if not a_fakequantize:
        a_fakequantize = backend_params["default_act_quantize"]
    a_fakeq_params = extra_qparams.get("a_fakeq_params", {})
    # Get default observer type.
    if not w_observer:
        w_observer = backend_params["default_weight_observer"]
    if not a_observer:
        a_observer = backend_params["default_act_observer"]

    # Create qconfig.
    # here, rewrited by with_args
    w_qconfig = w_fakequantize.with_args(
        observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params()
    )
    a_qconfig = a_fakequantize.with_args(
        observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params()
    )
    logger.info(
        "Weight Qconfig:\n    FakeQuantize: {} Params: {}\n"
        "    Oberver:      {} Params: {}".format(
            w_fakequantize.__name__, w_fakeq_params, w_observer.__name__, str(w_qscheme)
        )
    )
    logger.info(
        "Activation Qconfig:\n    FakeQuantize: {} Params: {}\n"
        "    Oberver:      {} Params: {}".format(
            a_fakequantize.__name__, a_fakeq_params, a_observer.__name__, str(a_qscheme)
        )
    )
    if backend_params["qtype"] == "vitis":
        logger.info("Bias Qconfig:\n    TqtFakeQuantize with MinMaxObserver")

    return QConfig(activation=a_qconfig, weight=w_qconfig)


class CustomedTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """

    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        return m.__module__.startswith("torch.nn") and not isinstance(
            m, torch.nn.Sequential
        )


def duplicate_reused_nodes(graph: torch.fx.Graph, modules: Dict[str, Any] = {}):
    """graph是一个Graph而不是GraphModel,modules相当于用字典存放了所有的 named_modules:包括name和对应的Module"""
    _dup_prefix = "_dup"
    target_dict = dict()
    dup_modules = dict()
    for node in graph.nodes:
        if (
            node.op == "call_module"
        ):  # 找到调用模型的节点(即找到node.target=='call_module')，并在target_dict:(dict(list))中加入这个节点            if node.target not in target_dict:
            if node.target not in target_dict:
                target_dict[node.target] = [node]  # 这里node.target指向一个Module
            else:
                target_dict[node.target].append(node)
    for key in target_dict:  # 对所有操作为'call_module'的  key:Module
        if len(target_dict[key]) > 1:
            for idx, node in enumerate(
                target_dict[key]
            ):  # 对所有调用这个Module的 node, 都复制一遍这个Module
                if idx == 0:
                    continue
                module = deepcopy(modules[node.target])
                node.target += _dup_prefix + str(
                    idx
                )  # 让node(它调用Module的节点), 的target += '_dup_+idx
                dup_modules[
                    node.target
                ] = module  # 存放被复制的Module,放在字典dup_modules中(key是新的target名字,value就是这个新复制的Module)
    graph.lint()  # 检查Graph，比方说归属问题，拓扑排序这些
    return (
        graph,
        dup_modules,
    )  # 返回graph 和对应的 dup_modules(和modules对应都是存放Module的name作为key，对应的Moduleu作为value)


def prepare_constant_dict(graph: torch.fx.Graph, model: torch.nn.Module):
    """model是原来的模型,graph是经过trace后生成的Graph"""

    def _get_attrs(target, attrs):
        """辅助迭代函数,不断的从顶向下访问需要的属性.首先将attrs按照.分开,
        然后从target(target一开始等于model,如果把一个model看作是一个树的话,model相当于根节点)开始按照attrs访问attr,同时更新当前target
        """
        attrs = attrs.split(".")
        for att in attrs:
            target = getattr(target, att)
        return target

    constant_dict = dict()
    for node in graph.nodes:
        if node.op == "get_attr":
            constant_dict[node.target] = _get_attrs(
                model, node.target
            )  # 根据target指向的属性，来迭代访问属性.并使用(k=node.target,v=具体的属性变量)来保存存储。
    return constant_dict


def prepare_by_platform(
    model: torch.nn.Module,
    deploy_backend: BackendType,
    prepare_custom_config_dict: Dict[str, Any] = {},
    custom_tracer: Tracer = None,
):
    """
    Args:
        model (torch.nn.Module):
        deploy_backend (BackendType):

    >>> prepare_custom_config_dict : {
            extra_qconfig_dict : Dict, Find explanations in get_qconfig_by_platform,
            extra_quantizer_dict: Extra params for quantizer.
            preserve_attr: Dict, Specify attribute of model which should be preserved
                after prepare. Since symbolic_trace only store attributes which is
                in forward. If model.func1 and model.backbone.func2 should be preserved,
                {"": ["func1"], "backbone": ["func2"] } should work.
            Attr below is inherited from Pytorch.
            concrete_args: Specify input for model tracing.
            extra_fuse_dict: Specify extra fusing patterns and functions.
        }

    """
    model_mode = "Training" if model.training else "Eval"
    logger.info("Quantize model Scheme: {} Mode: {}".format(deploy_backend, model_mode))

    # Get Qconfig
    extra_qconfig_dict = prepare_custom_config_dict.get("extra_qconfig_dict", {})
    qconfig = get_qconfig_by_platform(deploy_backend, extra_qconfig_dict)

    _swap_ff_with_fxff(model)  # 单纯的将一些FloatFunctional操作替换为FXFloatFunctional
    # Preserve attr. 保留属性
    preserve_attr_dict = dict()
    if "preserve_attr" in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict["preserve_attr"]:
            cur_module = model
            if submodule_name != "":
                cur_module = getattr(model, submodule_name)
            preserve_attr_list = prepare_custom_config_dict["preserve_attr"][
                submodule_name
            ]
            preserve_attr_dict[submodule_name] = {}
            for attr in preserve_attr_list:
                preserve_attr_dict[submodule_name][attr] = getattr(cur_module, attr)
    # Symbolic trace 自定义trace，从concrete_args传trace的参数，leaf_module传不需要trace的模型
    concrete_args = prepare_custom_config_dict.get("concrete_args", None)
    customed_leaf_module = prepare_custom_config_dict.get("leaf_module", [])
    tracer = CustomedTracer(customed_leaf_module=tuple(customed_leaf_module))
    if custom_tracer is not None:
        tracer = custom_tracer
    graph = tracer.trace(model, concrete_args)
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    modules = dict(model.named_modules())
    graph, duplicated_modules = duplicate_reused_nodes(
        graph, modules
    )  # 复制存在重复用的调用Module的Node(),并且对调用这个Module的nodes每一个复制一份Module并让node.target指向这个新Module
    constant_nodes = prepare_constant_dict(
        graph, model
    )  # 对所有getattr (node.op='getattr') 节点，用constant_nodes提前访问好这些属性,并也用(name,attrbutation)的方式进行存储访问
    modules.update(duplicated_modules)  # 将duplicated_modules和constant_nodes融入到
    modules.update(constant_nodes)
    graph_module = GraphModule(
        modules, graph, name
    )  # GraphModule第一个参数可以是一个Module也可以是一个dict(存放了(str、对应的值),还有类名). GraphModule也有modules
    # Model fusion. 模型融合
    extra_fuse_dict = prepare_custom_config_dict.get("extra_fuse_dict", {})
    extra_fuse_dict.update(fuse_custom_config_dict)
    # Prepare
    import mqbench.custom_quantizer  # noqa: F401  在导入的时候就会执行装饰器函数

    extra_quantizer_dict = prepare_custom_config_dict.get("extra_quantizer_dict", {})
    quantizer = DEFAULT_MODEL_QUANTIZER[deploy_backend](
        extra_quantizer_dict, extra_fuse_dict
    )  # 使用 当前Backend对应的ModelQuantizer并传入额外的量化配置字典和融合字典来生成quantizer
    # print(quantizer)
    prepared = quantizer.prepare(graph_module, qconfig)
    # Restore attr. # 虽然可以恢复属性，但是在网络的更新中应该不能更新这些属性
    if "preserve_attr" in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict["preserve_attr"]:
            cur_module = prepared
            _type = type(model)
            if submodule_name != "":
                cur_module = getattr(prepared, submodule_name)
                _type = type(getattr(model, submodule_name))
            preserve_attr_list = prepare_custom_config_dict["preserve_attr"][
                submodule_name
            ]
            for attr_name in preserve_attr_list:
                logger.info("Preserve attr: {}.{}".format(submodule_name, attr_name))
                _attr = preserve_attr_dict[submodule_name][attr_name]
                if inspect.ismethod(_attr):
                    _attr = types.MethodType(getattr(_type, attr_name), cur_module)
                setattr(cur_module, attr_name, _attr)
    return prepared
