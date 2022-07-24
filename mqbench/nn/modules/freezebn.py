import torch

class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(*args, **kwargs)
        self.training = False

    def train(self, mode=False): # 重写了train的方法，从而即使执行了trainBN也不会变成train的模式
        self.training = False
        for module in self.children():
            module.train(False)
        return self