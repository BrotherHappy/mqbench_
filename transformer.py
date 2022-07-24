import mqbench,torch,torchvision,numpy as np,matplotlib.pyplot as plt,torchvision,torchvision.models as models,mmcv,mmcls,timm,timm.models as models
from torchmetrics import ConfusionMatrix
from mqbench.prepare_by_platform import prepare_by_platform
from mqbench.prepare_by_platform import BackendType
from mqbench.utils.state import enable_calibration
from mqbench.utils.state import enable_quantization
from mqbench.convert_deploy import convert_deploy
from tqdm import tqdm
from mmcls.datasets.builder import build_dataset,build_dataloader
from mmcls.datasets.builder import build_dataset,build_dataloader
device = torch.device('cpu')
mean=np.array([123.675, 116.28, 103.53])/255
std=np.array([58.395, 57.12, 57.375])/255

dataset_cfg = mmcv.Config.fromfile('/home/brother/Desktop/od/mmclassification/configs/_base_/datasets/imagenet_bs32.py')
dataset = build_dataset(dataset_cfg.data.val)
dataloader = build_dataloader(dataset=dataset,samples_per_gpu=4,workers_per_gpu=8,dist=False,shuffle=True,round_up=False)
print(f"length of dataset:{len(dataset)}")

model = timm.create_model('swin_base_patch4_window7_224',pretrained=True) #
backend = BackendType.Tensorrt
model = prepare_by_platform(model,backend) # trace模型，并且为Tensorrt Backend增加量化结点

enable_calibration(model) # 打开校准，准备好获取数据
#evaluation loop
cm = ConfusionMatrix(1000)
model.to(device)
for i,data in enumerate(tqdm(dataloader)):
    img = data['img'].to(device)
    model(img)
pass