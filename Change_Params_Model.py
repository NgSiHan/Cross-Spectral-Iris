from dataset import create_dataloader
import torch
from utils import *
from model import *
import torch.optim as optim

# Change the parameters to make fully connected layer fit the current dataset for classifier
def Change_Params(args):

    train_VIS,_,_,_,train_orig=create_dataloader(args)
    if(args.transfer==True):
         output_shape_orig = len(train_orig.classes*2)
    output_shape_new = len(train_VIS.classes*2)

    if(args.transfer==True):
        ResNet_Model=ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=output_shape_orig).to(device)
    else:
        ResNet_Model=ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=output_shape_new).to(device)
    print(ResNet_Model.eval())

    print("OUTPUT: ",output_shape_new)
    num_ftrs = ResNet_Model.fc.in_features

    # Need to adjust the output features to be compatible with new model (if transfered from different dataset)
    if(args.transfer==True):
        ResNet_Model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=num_ftrs, 
                        out_features=output_shape_orig, # same number of output units as our number of classes
                        bias=True)).to(device)
        optimizer = optim.Adam(ResNet_Model.parameters(),lr=1e-4)  
        state=torch.load(args.model_name+'.pt')
        ResNet_Model.load_state_dict(state["classifier"])
    
    # Change out_features to new shape (based on new dataset classes)
    ResNet_Model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=num_ftrs, 
                        out_features=output_shape_new, # same number of output units as our number of classes
                        bias=True)).to(device)
    optimizer = optim.Adam(ResNet_Model.parameters(),lr=1e-4)
    return ResNet_Model,output_shape_new,optimizer