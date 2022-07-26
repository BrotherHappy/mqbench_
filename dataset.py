import torch,PIL.Image as Image,numpy as np,torchvision.transforms as transforms,os,glob,os.path as osp
import random
import cv2 as cv
from torch.utils.data import Dataset,DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class ImageDataset(Dataset):
    def __init__(self,root='/data/datasets/imagenet/val'):
        super().__init__()
        # sorted_dirs = sorted(os.listdir(osp.join(root,'val')))
        sorted_dirs = sorted(os.listdir(root))
        map = {t:i for t,i in zip(sorted_dirs,range(len(sorted_dirs)))}
        img_paths = glob.glob(osp.join(root,r'*/*.JPEG'),recursive=True)
        self.ann =[(path,map[path.rsplit(r'/')[-2]]) for path in img_paths]
        # transform
        self.transform = transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((224,224)),transforms.ToTensor() \
        ,transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self,idx):
        # img = np.array(Image.open(self.ann[idx][0]))
        img = cv.imread(self.ann[idx][0])
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        if img.ndim != 3:
            img = np.stack([img,img,img],axis=-1)
        try:
            img = self.transform(img)
        except Exception as e:
            print('transforme err',self.ann[idx])
        
        return img,self.ann[idx][1]

class Dataset(Dataset):
    def __init__(self,root,transform=None):

        pass
    def __len__(self):
        pass
    def __getitem__(self,idx):
        pass

def get_dataloader():
    return DataLoader(dataset = ImageDataset(),batch_size=4,
    shuffle=True,num_workers=4)

if __name__=="__main__":
    dataset = ImageDataset()
    pass