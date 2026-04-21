import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import random
import numpy as np

from torchvision import datasets
from torchvision import transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]

#标准增强
train_trans_standard=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),#随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),#数据增强：50%概率水平翻转
    transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),#数据增强：颜色变换
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])

#无增强
train_trans_none=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),#图片大小为224x224
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])

#弱增强
train_trans_weak=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),#小旋转+平移
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])

#强增强
train_trans_strong=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=25),#大角度旋转
    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
    transforms.RandomErasing(p=0.2,scale=(0.02,0.2))
    ])
    

val_test_trans=train_trans_none

def get_loader(strategy,batch_size=32,num_workers=2):
    trans_dict={
        'none':train_trans_none,
        'standard':train_trans_standard,
        'weak':train_trans_weak,
        'strong':train_trans_strong
    }
    train_dataset= datasets.ImageFolder(root='data_split/train', transform=trans_dict[strategy])
    val_dataset= datasets.ImageFolder(root='data_split/val', transform=val_test_trans)
    test_dataset= datasets.ImageFolder(root='data_split/test',transform=val_test_trans)

    #创建Dataloader
    train_loader=DataLoader(train_dataset,batch_size,shuffle=True,num_workers=num_workers,persistent_workers=True)
    val_loader= DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True)
    test_loader= DataLoader(test_dataset,batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True)
    
    return train_loader,test_loader,val_loader


set_seed(42)
train_loader,test_loader,val_loader=get_loader('standard')
