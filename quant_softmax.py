import mqbench,torch,torchvision,numpy as np,matplotlib.pyplot as plt,torchvision,torchvision.models as models,timm,timm.models as models,torch.nn as nn
from torchmetrics import ConfusionMatrix
from plot import plot2dicts
from mqbench.prepare_by_platform import prepare_by_platform
from mqbench.prepare_by_platform import BackendType
from mqbench.utils.state import enable_calibration
from mqbench.utils.state import enable_quantization
from mqbench.convert_deploy import convert_deploy
from tqdm import tqdm
from mqbench.utils.logger import logger as log
from mqbench.fake_quantize.lsq import LearnableFakeQuantize
from dataset import get_dataloader
from timm.models.swin_transformer import SwinTransformer
from logger import get_logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
from torch.fx.graph_module import GraphModule
import timm.models.swin_transformer as st
from timm.models.swin_transformer import MySoftmax
print(DEFAULT_MODEL_QUANTIZER)
logger,workdir = get_logger("softmax_quant")
log = logger # 

device = torch.device('cuda')
st.M = st.M.to(device)
mean=np.array([123.675, 116.28, 103.53])/255
std=np.array([58.395, 57.12, 57.375])/255

# dataloader = get_dataloader() #
extra_qconfig_dict = {
    'w_observer': 'MinMaxObserver',
    'a_observer': 'EMAMinMaxObserver',
    # 'a_observer': 'MSEObserver',
    'w_fakequantize': 'FixedFakeQuantize',
    'a_fakequantize': 'FixedFakeQuantize',
    # 'a_fakequantize': 'LearnableFakeQuantize',
    # 'a_fakequantize': 'FixedFakeQuantize',
    'w_qscheme': {
        'bit': 8,
        'symmetry': True,
        'per_channel': True,
        'pot_scale': False
    },
    'a_qscheme': {
        'bit': 8,
        'symmetry': True,
        'per_channel': False,
        'pot_scale': False
    },
}
extra_quantizer_dict = {
    'additional_module_type':(MySoftmax,)
}
logger.info(extra_qconfig_dict) #  
model = timm.create_model('swin_base_patch4_window7_224',pretrained=True).to(device) # 创建模型
# model = timm.create_model('resnet18',pretrained=True).to(device) # 创建模型
prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict,'leaf_module':[MySoftmax],'extra_quantizer_dict':extra_quantizer_dict}
model = prepare_by_platform(model, BackendType.Tensorrt,prepare_custom_config_dict).to(device)
ori= timm.create_model('swin_base_patch4_window7_224',pretrained=True).to(device) #
model.eval() # 进行PTQ
enable_calibration(model) # 打开校准
dataloader = get_dataloader(shuffle=False) #
with torch.no_grad():
    for i,(img,label) in enumerate(tqdm(dataloader)):
        if i>=32:
            break
        img = img.to(device)
        model(img)

enable_quantization(model) # 打开量化，准备好模拟后台推断的量化
ori.to(device)
ori.eval()
from torchmetrics import Accuracy

from collections import defaultdict
d_ori = defaultdict(list)
d_quant = defaultdict(list)
def hook_generator(d,name):
    def hook(module,input,output):
        d[name].append((input[0].detach().cpu().numpy(),output.detach().cpu().numpy()))
        # d[name+'_output'].append(output.detach().cpu().numpy())
        return output
    return hook

# for name,m in ori.named_modules():
#     if name.endswith("softmax"):
#         m.register_forward_hook(hook_generator(d_ori,name))

# for name,m in model.named_modules():
#     if name.endswith("softmax"):
#         m.register_forward_hook(hook_generator(d_quant,name))

# model = ori
acc = Accuracy()
acc_ori = Accuracy()
with torch.no_grad():
    for i,(img,label) in enumerate(tqdm(dataloader)):
        # if i>10:
        #     break
        img = img.to(device)
        # acc.update(torch.argmax(model(img),dim=-1),torch.argmax(ori(img),dim=-1))
        acc.update(model(img).detach().cpu(),label.cpu())
        acc_ori.update(ori(img).detach().cpu(),label.cpu())
# plot2dicts(d_ori,d_quant)
logger.info(f"最终的精度是：{acc.compute()}")
logger.info(f"ori_最终的精度是：{acc_ori.compute()}")