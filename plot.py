import matplotlib.pyplot as plt,numpy as np,torch,torch.nn as nn,torchvision,os.path as osp
import matplotlib as mb

def plot2dicts(d1,d2,labels=("in","out")):# dict中每个键对应的值 是一个元组列表
    keys = d1.keys()
    c1 = 'green'
    c2 = 'blue'
    n = len(d1[next(iter(keys))][0])
    if (not labels) or len(labels)<n:
        labels = "None"*n
    for key in keys:
        fig:mb.figure.Figure= plt.figure(figsize=(20,10))
        axis = fig.subplots(1,n)

        for i,axi in enumerate(axis):
            axi.hist(np.concatenate([t[i] for t in d2[key]]).flatten(),bins = 256,label=labels[i]+"_quant",color=c2) # 
            axi.hist(np.concatenate([t[i] for t in d1[key]]).flatten(),bins=256,label=labels[i]+"_ori",color=c1,alpha=0.5)
            axi.legend()
        plt.savefig(osp.join('imgs',key+'.png'))
        # fig.legend()
        plt.pause(0.1)
        plt.show()
