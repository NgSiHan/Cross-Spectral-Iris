import glob
import os
from dataset import get_dataset
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
from model import *
import os
import cv2
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from Change_Params_Model import Change_Params
from src.matchers.arciris_network import iresnet100 as ArcIResNet100

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def str2bool(v):
    return str(v).lower() in ('true', '1', 'yes')

def save_img(img, img_path):
    cv2.imwrite(img_path, img)

inv_normalize = transforms.Normalize(
    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
    std=[1/0.5, 1/0.5, 1/0.5]
)

# Choose which datasets you would like to train. 
datasets_to_run=["Dataset"]
for data in datasets_to_run:
    parser = argparse.ArgumentParser(description='Coupled-GAN')
    # General
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')        
    parser.add_argument('--epochs', default=500, type=int, help='Number of epochs to run the model')
    parser.add_argument('--train', default=True, type=str2bool, help='Train=True,Test=False')
    parser.add_argument('--save_folder', type=str,default='./checkpoint/',help='path to save the checkpoints')
    parser.add_argument('--transfer',default=False, type=str2bool, help='if you would like to transfer num_classes from previous model')
    parser.add_argument('--linux',default=False, type=str2bool, help='if linux system is running the code (True) or Windows(False)')
    parser.add_argument('--verbose',default=True, type=str2bool, help='if you would to log output images periodically during training')
    # model setup
    parser.add_argument('--gan_loss', default=1e-3, type=float, help='gan loss HyperParameter')
    parser.add_argument('--l1_loss', default=1, type=float, help='l2 loss HyperParameter')
    parser.add_argument('--rec_loss', default=1, type=float, help='recognition loss HyperParameter')
    parser.add_argument('--per_loss', default=1e-1, type=float, help='perceptual loss HyperParameter')
    parser.add_argument('--update_ratio', default=2, type=float, help='How many times should generator update before discriminator')
    parser.add_argument('--classifier_pretrain', default=0, type=float, help='How long to let the classifier pretrain before generator, 0 if none')
    parser.add_argument('--patch_gan',default=True, type=str2bool, help='Use patch-gan (True) or regular GAN (False)')
    parser.add_argument('--relativistic', default=True, type=str2bool, help='Relativistic (True) or normal discriminator (False)')
    parser.add_argument('--use_lr_decay', default=False, type=str2bool, help='Use lr decay (True) or not (False)')
    parser.add_argument('--three_player', default=True, type=str2bool, help='Three player game with classifier (True), or allow classifier to be spectator (False)')
    # Dataset
    parser.add_argument('--modality',default='normalized',type=str,help='[normalized] or [cropped] iris images')
    parser.add_argument('--VIS_folder', type=str,default='./'+str(data)+'/VIS',help='path to VIS')
    parser.add_argument('--NIR_folder', type=str,default='./'+str(data)+'/NIR',help='path to NIR')
    parser.add_argument('--continue_old', default=False, type=str2bool, help='Continue a model (model_name must exist if True)')
    parser.add_argument('--model_name', type=str,default="model_resnet18_classifier_"+str(data),help='path to save the data')
    parser.add_argument('--start_epoch', default=0, type=int,help='Start Epoch for continuing')
    parser.add_argument('--save_every', default=25, type=int, help='Save a numbered backup checkpoint every N epochs (0 = disable) [default 25]')
    # ArcIris identity loss — directly optimises the generator for the evaluation metric
    parser.add_argument('--arciris_model', type=str, default='./models-ArcIris/ResNet100_154000.pt',
                        help='Frozen ArcIris iresnet100 checkpoint used for identity loss [default: ./models-ArcIris/ResNet100_154000.pt]')
    parser.add_argument('--arciris_loss', default=1.0, type=float,
                        help='ArcIris identity loss weight applied to G_VIS→NIR (0 = disabled) [default 1.0]')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    netR,out_shape,optimizer_C_1=Change_Params(args)

    if(args.modality=="normalized"):    
        net_1=UNet(feat_dim=128)
        net_2=UNet(feat_dim=128)
    else:
        net_1=UNet(feat_dim=256)
        net_2=UNet(feat_dim=256)
    if(args.patch_gan==True):
        disc_1 = NLayerDiscriminator(input_nc=3)
        disc_2 = NLayerDiscriminator(input_nc=3)
    else:
        disc_1 = Discriminator(in_channels=3)
        disc_2 = Discriminator(in_channels=3)

    # Generators
    net_1.to(device)
    net_1.train()

    net_2.to(device)
    net_2.train()
    # If loading from checkpoint, load dictionaries, start epoch
    if(args.continue_old==True):
        ckpt_path = args.save_folder + args.model_name + '.pt'
        state=torch.load(ckpt_path)
        disc_1.load_state_dict(state["disc_1"])
        disc_2.load_state_dict(state["disc_2"])
        net_1.load_state_dict(state["net_1"])
        net_2.load_state_dict(state["net_2"])
        netR.load_state_dict(state["classifier"])
        start_epoch=state["epoch"]+1
        print(f'Resuming from {ckpt_path}  (epoch {state["epoch"]} -> continuing at {start_epoch})')
    else:
        state=None
        start_epoch=0

    #Discriminators
    disc_1.to(device)
    disc_1.train()

    disc_2.to(device)
    disc_2.train()

    # ArcIris frozen identity-loss model (G_VIS→NIR only)
    # Directly optimises fake-NIR to match real-NIR in ArcIris embedding space —
    # the exact metric used in 05_evaluate.py. Weights are frozen; only the
    # generator's weights are updated through this loss.
    if args.arciris_loss > 0:
        _arc_path = args.arciris_model
        if not os.path.exists(_arc_path):
            raise FileNotFoundError(
                f'ArcIris model not found: {_arc_path}\n'
                'Pass --arciris_loss 0 to disable the identity loss.')
        netArc = ArcIResNet100(pretrained=False)
        _arc_sd = torch.load(_arc_path, map_location=device)
        _arc_sd = {k.replace('module.', ''): v for k, v in _arc_sd.items()}
        netArc.load_state_dict(_arc_sd, strict=True)
        netArc = netArc.to(device).eval()
        for p in netArc.parameters():
            p.requires_grad = False
        print(f'ArcIris identity loss ENABLED  weight={args.arciris_loss}  '
              f'model={_arc_path}')
    else:
        netArc = None
        print('ArcIris identity loss DISABLED  (--arciris_loss 0)')

    # Set optimizers
    optimizer_G_1 = torch.optim.Adam(list(net_1.parameters()), lr=1e-4,betas=(0.9,0.999), weight_decay=1e-4)
    optimizer_G_2 = torch.optim.Adam(list(net_2.parameters()), lr=1e-4,betas=(0.9,0.999), weight_decay=1e-4)
    optimizer_D_1 = torch.optim.Adam(list(disc_1.parameters()), lr=1e-4,betas=(0.9,0.999), weight_decay=1e-4)
    optimizer_D_2 = torch.optim.Adam(list(disc_2.parameters()), lr=1e-4,betas=(0.9,0.999), weight_decay=1e-4)
    
    # Restore optimizer states when continuing (must be after optimizers are created)
    if state is not None:
        optimizer_G_1.load_state_dict(state["optimizer_1"])
        optimizer_G_2.load_state_dict(state["optimizer_2"])
        if "optimizer_D_1" in state:
            optimizer_D_1.load_state_dict(state["optimizer_D_1"])
            optimizer_D_2.load_state_dict(state["optimizer_D_2"])
        if "optimizer_C_1" in state:
            optimizer_C_1.load_state_dict(state["optimizer_C_1"])

    # Scheduler only used if args.use_lr_decay = True
    decay_rate=0.96
    G_1_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_G_1,gamma=decay_rate)
    G_2_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_G_2,gamma=decay_rate)
    D_1_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D_1,gamma=decay_rate)
    D_2_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_D_2,gamma=decay_rate)

    # If classifier will be competing with generator
    if(args.three_player==True):
        C_1_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_C_1,gamma=decay_rate)

    # Initialize loss functions and networks 
    adversarial_loss = torch.nn.MSELoss().to(device)
    L1_Norm_loss = torch.nn.L1Loss().to(device)
    L2_Norm_loss = torch.nn.MSELoss().to(device)
    cosine_loss=nn.CosineEmbeddingLoss().to(device)
    netF=FeatureExtractor(vgg19(weights=VGG19_Weights.DEFAULT)).to(device)
    ceLoss=nn.CrossEntropyLoss().to(device)

    train_loader = get_dataset(args) 
    print(len(train_loader))

    Tensor = torch.cuda.FloatTensor
    torch.cuda.empty_cache()
    for epoch in range(start_epoch,args.epochs):

        print(epoch)
        # Variables for updating losses
        loss_m_d_1 = AverageMeter()
        loss_m_g_1 = AverageMeter()
        loss_m_d_2 = AverageMeter()
        loss_m_g_2 = AverageMeter()
        loss_m_r_1 = AverageMeter()
        loss_m_r_2 = AverageMeter()
        loss_m_p_1 = AverageMeter()
        loss_m_p_2 = AverageMeter()
        loss_m_class = AverageMeter()
        loss_m_a_1 = AverageMeter()
        loss_m_a_2 = AverageMeter()
        loss_m_arc_2 = AverageMeter()
        iteration=0

        for iter, (img_1, img_2, lbl,cls1,cls2,id) in enumerate(train_loader):

            loss_Fake_2=0
            loss_Fake_1=0
            true_label_1=[]
            true_label_2=[]

            for i in range(len(cls2)):
                true_label_1.append((F.one_hot(torch.LongTensor([cls1[i]]), num_classes=out_shape)))
                true_label_2.append((F.one_hot(torch.LongTensor([cls2[i]]), num_classes=out_shape)))

            true_class_1 = torch.cat(true_label_1,dim=0).to(torch.float).to(device) 
            true_class_2 = torch.cat(true_label_2,dim=0).to(torch.float).to(device)

            bs = img_1.size(0)
            img_1, img_2, lbl = img_1.to(device), img_2.to(device), lbl.type(torch.float).to(device)

            if(args.patch_gan==True):
                if(args.modality=='normalized'):
                    valid = Variable(torch.rand(bs, 1,6,62) * 0.5 + 0.7,requires_grad=False).to(device)
                    fake = Variable(torch.rand(bs, 1,6,62) * 0.3,requires_grad=False).to(device)
                elif(args.modality=="cropped"):
                    valid = Variable(torch.rand(bs, 1,30,30) * 0.5 + 0.7,requires_grad=False).to(device)
                    fake = Variable(torch.rand(bs, 1,30,30) * 0.3,requires_grad=False).to(device)
            else:
                valid = Variable(torch.rand(bs, 1) * 0.5 + 0.7,requires_grad=False).to(device)
                fake = Variable(torch.rand(bs, 1) * 0.3,requires_grad=False).to(device)

            ################ First generator ################################################
            fake_2,_ = net_1(img_1)
            # Recognition loss from classifier
            if(args.rec_loss>0):
                if(args.three_player==True):
                    # Only update according to update ratio
                    if(iteration%args.update_ratio==0):
                        rec_fake_2,_=netR(fake_2)
                        rec_real_2,_=netR(img_2)
                        # If classifier past pretrain, add the fake iris classification into classification loss
                        if(epoch>=args.classifier_pretrain):
                            loss_class_2=0.4*ceLoss(rec_fake_2,true_class_1)+ceLoss(rec_real_2,true_class_2)
                        # Otherwise, only update classifier according to real images
                        else:
                            loss_class_2=ceLoss(rec_real_2,true_class_2)
                        # backprop classifier model
                        optimizer_C_1.zero_grad()
                        loss_class_2.backward()
                        optimizer_C_1.step()

                # Update recognition loss based on cosine loss of real and generated images
                fake_2,_ = net_1(img_1)
                _,rec_fea_real_2=netR(img_2)
                _,rec_fea_fake_2=netR(fake_2)
                loss_rec_2=cosine_loss(rec_fea_fake_2,rec_fea_real_2,torch.ones(1).to(device))
                loss_Fake_2+=loss_rec_2.mean() * args.rec_loss
                loss_m_r_2.update(loss_rec_2.item())
            # Perceptual loss for high-level features in image, L2 norm loss used 
            if(args.per_loss>0):
                fea_2=netF(img_2)
                fea_fake_2=netF(fake_2.to(device))
                loss_perc_2=L2_Norm_loss(fea_2,fea_fake_2)
                loss_Fake_2+=loss_perc_2*args.per_loss
                loss_m_p_2.update(loss_perc_2.item())
            pred_fake_2 = disc_2(fake_2)
            pred_real_2 = disc_2(img_2).detach()
            # Relativistic GAN (https://arxiv.org/abs/1807.00734) or standard GAN
            if(args.relativistic==True):
                loss_GAN_Fake_2 = adversarial_loss(pred_fake_2 - torch.mean(pred_real_2), valid)
            else:
                loss_GAN_Fake_2 = adversarial_loss(pred_fake_2, valid)
            # L1 loss
            loss_L1_Fake_2 = (L1_Norm_loss(fake_2, img_2))
            # Update loss
            loss_Fake_2 +=  loss_GAN_Fake_2 * args.gan_loss + loss_L1_Fake_2 * args.l1_loss
            # ArcIris identity loss: push fake NIR to match real NIR in ArcIris embedding space.
            # Gradients flow through fake_2 → net_1; netArc is frozen (no param update).
            if netArc is not None and args.arciris_loss > 0:
                with torch.no_grad():
                    arc_real = F.normalize(netArc(img_2), dim=1)          # target, detached
                # Match evaluation pipeline exactly:
                # 04_generate_fake_nir.py averages the 3 UNet output channels to a
                # single-channel greyscale before saving; 05_evaluate.py then repeats
                # that channel to 3 identical channels before calling ArcIris.
                # If we skip this averaging here, ArcIris sees 3 independent channels
                # during training but 3 identical channels at evaluation — a mismatch
                # that causes the training Arc2 loss to underestimate evaluation difficulty.
                fake_2_grey = fake_2.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
                arc_fake = F.normalize(netArc(fake_2_grey), dim=1)        # gradient: fake_2_grey → fake_2
                loss_arc_2 = (1.0 - (arc_fake * arc_real).sum(dim=1)).mean()
                loss_Fake_2 += loss_arc_2 * args.arciris_loss
                loss_m_arc_2.update(loss_arc_2.item())

            # If classifier is finished pretraining, update Generator as well
            if(epoch>=args.classifier_pretrain):
                optimizer_G_1.zero_grad()
                loss_Fake_2.backward(retain_graph=True)
                optimizer_G_1.step()
            # Update the values
            loss_m_g_2.update(loss_Fake_2.item())
            loss_m_a_2.update(loss_GAN_Fake_2.item())

            ################ Second generator ################################################
            fake_1,_ = net_2(img_2)
            
            if(args.rec_loss>0):
                if(args.three_player==True):
                    # Only update according to update ratio
                    if(iteration%args.update_ratio==0):
                        rec_real_1,_=netR(img_1)
                        rec_fake_1,_=netR(fake_1)
                        # If classifier past pretrain, add the fake iris classification into classification loss
                        if(epoch>=args.classifier_pretrain):
                            loss_class_1=0.4*ceLoss(rec_fake_1,true_class_2)+ceLoss(rec_real_1,true_class_1)
                        # Otherwise, only update classifier according to real images
                        else:
                            loss_class_1=ceLoss(rec_real_1,true_class_1)
                        # backprop classifier model
                        optimizer_C_1.zero_grad()
                        loss_class_1.backward()
                        optimizer_C_1.step()
                        loss_m_class.update((loss_class_1.item()+loss_class_2.item())/2)

                # Update recognition loss based on cosine loss of real and generated images
                fake_1,_ = net_2(img_2)
                _,rec_fea_real_1=netR(img_1)
                _,rec_fea_fake_1=netR(fake_1)
                loss_rec_1=cosine_loss(rec_fea_fake_1,rec_fea_real_1,torch.ones(1).to(device))  
                loss_Fake_1+=loss_rec_1.mean() * args.rec_loss
                loss_m_r_1.update(loss_rec_1.item())

            # Perceptual loss for high-level features in image, L2 norm loss used 
            if(args.per_loss>0):
                fea_1=netF(img_1)
                fea_fake_1=netF(fake_1.to(device))
                loss_perc_1=L2_Norm_loss(fea_1,fea_fake_1)
                loss_Fake_1+=loss_perc_1*args.per_loss
                loss_m_p_1.update(loss_perc_1.item())

            pred_fake_1 = disc_1(fake_1)
            pred_real_1 = disc_1(img_1).detach()
            # Relativistic GAN (https://arxiv.org/abs/1807.00734) or standard GAN
            if(args.relativistic==True):
                loss_GAN_Fake_1 = adversarial_loss(pred_fake_1 - torch.mean(pred_real_1), valid)
            else:
                loss_GAN_Fake_1 = adversarial_loss(pred_fake_1, valid)

            loss_L1_Fake_1 = (L1_Norm_loss(fake_1, img_1)) 
            loss_Fake_1 +=  loss_GAN_Fake_1 * args.gan_loss + loss_L1_Fake_1 * args.l1_loss

            if(epoch>=args.classifier_pretrain):
                optimizer_G_2.zero_grad()
                loss_Fake_1.backward(retain_graph=True)
                optimizer_G_2.step()
            loss_m_g_1.update(loss_Fake_1.item())
            loss_m_a_1.update(loss_GAN_Fake_1.item())

            if(iteration%args.update_ratio==0):
                ##################### Discriminators ##################################
                pred_real_1 = disc_1(img_1)
                pred_fake_1 = disc_1(fake_1.detach())
                
                pred_real_2 = disc_2(img_2)
                pred_fake_2 = disc_2(fake_2.detach())

                if(epoch>=args.classifier_pretrain):
                    if(args.relativistic==True):
                        d_loss_1 = (adversarial_loss(pred_fake_1-torch.mean(pred_real_1), fake) +
                                    adversarial_loss(pred_real_1-torch.mean(pred_fake_1), valid))/2
                        d_loss_2 = (adversarial_loss(pred_fake_2-torch.mean(pred_real_2), fake) +
                                    adversarial_loss(pred_real_2-torch.mean(pred_fake_2), valid))/2
                    else:
                        d_loss_1 = (
                            adversarial_loss(pred_real_1, valid)
                            + adversarial_loss(pred_fake_1, fake)
                            ) / 2
                        d_loss_2 = (
                            adversarial_loss(pred_real_2, valid)
                            + adversarial_loss(pred_fake_2, fake)
                            ) / 2
                    # Update optimizers
                    optimizer_D_1.zero_grad()
                    d_loss_1.backward()
                    optimizer_D_1.step()

                    optimizer_D_2.zero_grad()
                    d_loss_2.backward()
                    optimizer_D_2.step()

                    loss_m_d_1.update(d_loss_1.item())
                    loss_m_d_2.update(d_loss_2.item())
            
            lr=G_1_scheduler.get_last_lr()[0]
            if(args.verbose):
                if iteration % 200 == 0:
                    print('epoch: %02d, iter: %02d/%02d, lr: %.4f, D1 loss: %.4f, G1 loss: %.4f, A1 loss: %.4f, R1 loss: %.4f, P1 loss: %.4f, Arc2 loss: %.4f, D2 loss: %.4f, G2 loss: %.4f, A2 loss: %.4f, R2 loss: %.4f, P2 loss: %.4f, class loss: %.4f' % (
                        epoch, iteration, len(train_loader), lr,loss_m_d_1.avg, loss_m_g_1.avg, loss_m_a_1.avg, loss_m_r_1.avg,loss_m_p_1.avg, loss_m_arc_2.avg, loss_m_d_2.avg, loss_m_g_2.avg,loss_m_a_2.avg, loss_m_r_2.avg,loss_m_p_2.avg,loss_m_class.avg ))
                    if(epoch>=args.classifier_pretrain):
                        
                        if(args.modality=='normalized'):
                            path1="valid/normalized_resnet18_classifier_cl_"+str(data)
                        elif(args.modality=='cropped'):
                            path1="valid/cropped_resnet18_classifier_cl_"+str(data)
                        os.makedirs(path1, exist_ok=True)
                        vis_show  = inv_normalize(img_1.detach())
                        nir_show  = inv_normalize(img_2.detach())
                        fake_nir_show = inv_normalize(fake_2.detach())
                        fake_vis_show = inv_normalize(fake_1.detach())
                        save_img(tensor2img(nir_show[0].float().cpu()),      path1+'/' + str(epoch)+'_'+str(iteration) + '_RealNIR.png')
                        save_img(tensor2img(vis_show[0].float().cpu()),      path1+'/' + str(epoch)+'_'+str(iteration) + '_RealVIS.png')
                        save_img(tensor2img(fake_nir_show[0].float().cpu()), path1+'/' + str(epoch)+'_'+str(iteration) + '_FakeNIR.png')
                        save_img(tensor2img(fake_vis_show[0].float().cpu()), path1+'/' + str(epoch)+'_'+str(iteration) + '_FakeVIS.png')
            
            iteration+=1
        if(args.use_lr_decay==True):
                G_1_scheduler.step()
                D_1_scheduler.step() 
                G_2_scheduler.step()
                D_2_scheduler.step() 
                C_1_scheduler.step()

        state = {}

        state['net_1'] = net_1.state_dict()
        state['optimizer_1'] = optimizer_G_1.state_dict()
        state['disc_1'] = disc_1.state_dict()
        state['optimizer_D_1'] = optimizer_D_1.state_dict()
        state['classifier']=netR.state_dict()
        state['optimizer_C_1'] = optimizer_C_1.state_dict()
        state['net_2'] = net_2.state_dict()
        state['optimizer_2'] = optimizer_G_2.state_dict()
        state['disc_2'] = disc_2.state_dict()
        state['optimizer_D_2'] = optimizer_D_2.state_dict()
        state['epoch'] = epoch
        # Always overwrite the latest checkpoint (for resuming)
        os.makedirs(args.save_folder, exist_ok=True)
        latest_path = args.save_folder + args.model_name + '.pt'
        torch.save(state, latest_path)

        # Periodic numbered backup every --save_every epochs, keep last 3
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            backup_path = args.save_folder + args.model_name + f'_epoch_{epoch:04d}.pt'
            torch.save(state, backup_path)
            old_backups = sorted(glob.glob(
                args.save_folder + args.model_name + '_epoch_*.pt'))
            for old in old_backups[:-3]:
                os.remove(old)
            print(f'Backup checkpoint saved: {backup_path}')

        print('\nmodel saved!\n')