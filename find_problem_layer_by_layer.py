import mqbench,torch,torchvision,numpy as np,matplotlib.pyplot as plt,torchvision,torchvision.models as models,timm,timm.models as models
from torchmetrics import ConfusionMatrix
from mqbench.prepare_by_platform import prepare_by_platform
from mqbench.prepare_by_platform import BackendType
from mqbench.utils.state import enable_calibration
from mqbench.utils.state import enable_quantization
from mqbench.convert_deploy import convert_deploy
from tqdm import tqdm
from mqbench.utils.logger import logger as log
from mqbench.fake_quantize.lsq import LearnableFakeQuantize
from dataset import get_dataloader
from logger import get_logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
from torch.fx.graph_module import GraphModule
print(DEFAULT_MODEL_QUANTIZER)
logger,workdir = get_logger("SwinQuant-qkv+Conv+Linear(tensorrt-default)")
log = logger



device = torch.device('cuda:1')
mean=np.array([123.675, 116.28, 103.53])/255
std=np.array([58.395, 57.12, 57.375])/255


dataloader = get_dataloader()
extra_qconfig_dict = {
    'w_observer': 'MinMaxObserver',
    'a_observer': 'EMAMinMaxObserver',
    'w_fakequantize': 'FixedFakeQuantize',
    'a_fakequantize': 'LearnableFakeQuantize',
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
    }
}
logger.info(extra_qconfig_dict) #  

model = timm.create_model('swin_base_patch4_window7_224',pretrained=True).to(device) #
prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
# model = prepare_by_platform(model, BackendType.Tensorrt, prepare_custom_config_dict).to(device)
model = prepare_by_platform(model, BackendType.Tensorrt,prepare_custom_config_dict).to(device)
ori= timm.create_model('swin_base_patch4_window7_224',pretrained=True).to(device) #
model.eval() # 进行PTQ
enable_calibration(model) # 打开校准

# 校准
with torch.no_grad():
    for i,(img,label) in enumerate(tqdm(dataloader)):
        if i>=32:
            break
        img = img.to(device)
        model(img)

# 分别为model和ori 都注册模型
from collections import defaultdict
quant_dict = defaultdict(list)
ori_dict = defaultdict(list)
def hook_generator(d,name):
    def hook(module,input,output):
        d[name].append(input[0])
        return output
    return hook
for name,m in model.named_modules():
    if 'mid' in name:
        m.register_forward_hook(hook_generator(quant_dict,name))
for name,m in ori.named_modules():
    if 'mid' in name:
        m.register_forward_hook(hook_generator(ori_dict,name))

# 量化计算
enable_quantization(model) # 打开量化，准备好模拟后台推断的量化
ori.to(device)
ori.eval()
from torchmetrics import Accuracy

# model = ori
acc = Accuracy()
ori_acc = Accuracy()
with torch.no_grad():
    for i,(img,label) in enumerate(tqdm(dataloader)):
        if i>32:
            break
        img = img.to(device)
        # acc.update(torch.argmax(model(img),dim=-1),torch.argmax(ori(img),dim=-1))
        acc.update(model(img).detach().cpu(),label.cpu())
        ori_acc.update(ori(img).detach().cpu(),label.cpu())
logger.info(f"最终的精度是：{acc.compute()}")
logger.info(f"ori的精度:{ori_acc.compute()}")

