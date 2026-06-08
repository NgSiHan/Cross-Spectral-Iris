import os
from dataset import get_dataset
import argparse
# from utils import *
import torch
from utils import *
from model import *
from torchvision import transforms

#Helper functions#################################################################################
inv_normalize = transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5]
)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

################################################################################################
datasets_to_run=["Dataset"]
for data in datasets_to_run:
    modality_to_run=["VIS","NIR"]
    for mode in modality_to_run:
        parser = argparse.ArgumentParser(description='Coupled-GAN')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        parser.add_argument('--VIS_folder', type=str,default='./'+str(data)+'/VIS_Valid',help='path to VIS')
        parser.add_argument('--NIR_folder', type=str,default='./'+str(data)+'/NIR_Valid',help='path to NIR')
        parser.add_argument('--save_folder', type=str,default='./checkpoint/',help='path to save the data')
        parser.add_argument('--linux',default=False,help='if linux system is running the code (True) or Windows(False)')

        # model setup

        parser.add_argument('--modality',default='cropped',type=str,help='[normalized] or [cropped] iris images')
        parser.add_argument('--train', default=False, help='Train=True,Test=False')
        parser.add_argument('--checkpt_load',default="./checkpoint/model_resnet18_classifier_"+str(data)+".pt",help="path to checkpoint")
        parser.add_argument('--vistonir_dir',default="results/resnet18_classifier_cropped_vis_to_nir_"+str(data),help="path to output images visible to nir")
        parser.add_argument('--nirtovis_dir',default="results/resnet18_classifier_cropped_nir_to_vis_"+str(data),help="path to output images nir to visible")

        args = parser.parse_args()

        CUDA_VISIBLE_DEVICES = 0
        torch.cuda.memory_summary(device=None, abbreviated=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if(args.modality=="normalized"):
            net = UNet(feat_dim=128)
        elif(args.modality=="cropped"):
            net = UNet(feat_dim=256)
        state=torch.load(args.checkpt_load)
        if(mode=="VIS"):
            net.load_state_dict(state['net_1'])
            net.to(device)
            
        elif(mode=="NIR"):
            net.load_state_dict(state['net_2'])
            net.to(device)
        
        train_loader = get_dataset(args)
        print(len(train_loader))

        for iter, (img_1, img_2, lbl,_,_,id) in enumerate(train_loader):
            
            print(iter)
            bs = img_1.size(0)
            
            img_1, img_2= img_1.to(device), img_2.to(device)
                
            if(mode=="VIS"):
                fake_2, y_1 = net(img_1)
                path1=args.vistonir_dir
                os.makedirs(path1 , exist_ok=True)
                img_1=inv_normalize(img_1)
                img_2=inv_normalize(img_2)
                fake_2=inv_normalize(fake_2)
                
                save_img(tensor2img(img_2.detach()[0].float().cpu()), path1+ '/'+str(id).strip('[\']')+'_' + str(iter) + '_RealNIR.png')
                save_img(tensor2img(img_1.detach()[0].float().cpu()), path1+'/' +str(id).strip('[\']')+'_' + str(iter) + '_RealVIS.png')
                save_img(tensor2img(fake_2.detach()[0].float().cpu()), path1+'/' +str(id).strip('[\']')+'_' + str(iter) +'_FakeNIR.png')
            elif(mode=="NIR"):
                fake_1, y_2 = net(img_2)
                path2=args.nirtovis_dir
                os.makedirs(path2 , exist_ok=True)
                img_1=inv_normalize(img_1)
                img_2=inv_normalize(img_2)
                fake_1=inv_normalize(fake_1)
                save_img(tensor2img(img_2.detach()[0].float().cpu()), path2+ '/'  +str(id).strip('[\']')+'_' + str(iter) + '_RealNIR.png')
                save_img(tensor2img(img_1.detach()[0].float().cpu()), path2+'/' +str(id).strip('[\']')+'_' + str(iter) + '_RealVIS.png')
                save_img(tensor2img(fake_1.detach()[0].float().cpu()), path2+'/' +str(id).strip('[\']')+'_' + str(iter) +'_FakeVIS.png')
