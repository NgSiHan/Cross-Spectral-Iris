import random
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch
from utils import *
import numbers
import numpy as np
import torch
from torchvision.transforms.functional import pad
from torchvision.datasets import ImageFolder

    
class ContrastiveDataset(Dataset):
    def __init__(self, dataset_1, dataset_2, args,positive_prob=0.5):
        super().__init__()
        self.VIS = dataset_1
        self.NIR = dataset_2
        self.args=args
        self.positive_prob = positive_prob

        print(len(self.VIS))  
        print(len(self.NIR))

        self.positive_h = {}
        self.negative_h = {}

        for i in range(len(self.VIS)):
            # contruct the positive pair correspondence
            img_address = self.VIS.imgs[i][0]
            if(self.args.linux==True):
                id = img_address.split('/')[-2]
            else:
                id1 = img_address.split('/')
                id= id1[-1].split('\\')[-2]
            print(id)
            if id in self.positive_h:
                self.positive_h[id].append(i)
            else:
                self.positive_h[id] = [i]
            # construct the negative pair correspondence
            for j in range(len(self.VIS.imgs)):
                profile_address = self.VIS.imgs[j][0]
               
                if id in profile_address:
                    if id in self.negative_h:
                        self.negative_h[id].append(j)
                    else:
                        self.negative_h[id] = [j]

    def __getitem__(self, index):
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob
        img_0, label_0 = self.VIS[index]

        if(self.args.train==True):
            img_address = self.VIS.imgs[index][0]
            if(self.args.linux==True):
                id = img_address.split('/')[-2]
            else:
                id1 = img_address.split('/')
                id= id1[-1].split('\\')[-2]
            idx_neg = self.negative_h[id]

            rnd_idx = random.randint(0, len(idx_neg) - 1)
            idx_neg = idx_neg[rnd_idx]
            img_1, label_1 = self.NIR[idx_neg]

        else:
            img_address = self.VIS.imgs[index][0]
            if(self.args.linux==True):
                id = img_address.split('/')[-2]
            else:
                id1 = img_address.split('/')
                id= id1[-1].split('\\')[-2]
            img_1, label_1 = self.NIR[index]

        lbl_1=len(self.VIS.classes)+label_1
        return img_0, img_1,same_class,label_0,lbl_1,id

    def __len__(self):
        # return min(len(self.morph), len(self.photo))
        return len(self.NIR)

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding
class Pad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)

def get_dataset(args):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    resize = transforms.Resize(size=(256,
            256))
    if(args.train==True):
        if args.modality=="normalized":
            VIS_dataset = datasets.ImageFolder(
            args.VIS_folder,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                    std=std),
            ]))

            NIR_dataset = datasets.ImageFolder(
                args.NIR_folder,
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,
                                        std=std),
                ]))
        else:

            VIS_dataset = datasets.ImageFolder(
            args.VIS_folder,
            transforms.Compose([
                Pad(),resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                    std=std),
            ]))

            NIR_dataset = datasets.ImageFolder(
                args.NIR_folder,
                transforms.Compose([
                    Pad(),resize,
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean,
                                        std=std),
                ]))

        train_loader = torch.utils.data.DataLoader(
            ContrastiveDataset(VIS_dataset, NIR_dataset,args,positive_prob=1), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    else:
        VIS_dataset = datasets.ImageFolder(
            args.VIS_folder,
            transforms.Compose([
                Pad(), resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        )

        NIR_dataset = datasets.ImageFolder(
            args.NIR_folder,
            transforms.Compose([
                Pad(), resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        )
        

        train_loader = torch.utils.data.DataLoader(
            ContrastiveDataset(VIS_dataset, NIR_dataset,args,positive_prob=1), batch_size=1, shuffle=False, pin_memory=True)
    return train_loader
def create_dataloader(args):
    # initialize our data augmentation functions
    resize = transforms.Resize(size=(256,
            256))
    rotate = transforms.RandomRotation(degrees=(-15,15))
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),

    # initialize our training and validation set data augmentation
    # pipeline

    trainTransforms = transforms.Compose([rotate,transforms.ToTensor(),normalize])
    valTransforms = transforms.Compose([resize, transforms.ToTensor(),normalize])

    # initialize the training and validation dataset
    print("[INFO] loading the training and validation dataset...")
    trainVIS = ImageFolder(root=args.VIS_folder,
            transform=trainTransforms)
    valVIS = ImageFolder(root=args.VIS_folder, 
             transform=valTransforms)
    trainNIR = ImageFolder(root=args.NIR_folder,
            transform=trainTransforms)
    valNIR = ImageFolder(root=args.NIR_folder, 
            transform=valTransforms)
    if(args.transfer==True):
        transferVIS= ImageFolder(root=args.orig_folder, 
             transform=trainTransforms)
    else:
        transferVIS=None
    
    print("[INFO] training dataset contains {} samples...".format(
            len(trainVIS)))

    # create training and validation set dataloaders
    print("[INFO] creating training and validation set dataloaders...")


    # grab a batch from both training and validation dataloader

    # visualize the training and validation set batches
    print("[INFO] visualizing training and validation batch...")


    return trainVIS,trainNIR,valVIS,valNIR,transferVIS
