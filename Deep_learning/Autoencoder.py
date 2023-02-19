# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 19:20:14 2021

@author: 22649517
"""

from six.moves import urllib
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
print(os.getcwd())
from pathlib import Path
from torch.autograd import Variable
import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
import pytorch_ssim
from torchvision.utils import save_image


opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)



root_dir = 'C:/Users/'
train_X_dir = root_dir + '/train_X/' 
train_Y_dir = root_dir + '/train_Y/'
test_X_dir = root_dir + '/test_X/'
test_Y_dir = root_dir + '/test_Y/'




class dataset(Dataset):

    def __init__(self, x_dir, y_dir):      
        self.x_paths = []
        self.y_paths = []
        
        for f in Path(x_dir).glob("*.npy"):
            self.x_paths.append(f)

        for f in Path(y_dir).glob("*.npy"):
            self.y_paths.append(f)
          
        self.x_paths = sorted(self.x_paths) 
        self.y_paths = sorted(self.y_paths) 
            
        self.len = len(self.x_paths)
    
    def __getitem__(self, index):
        
        print(self.x_paths[index])
        self.x = np.load(self.x_paths[index]).astype(np.float64)
        self.y = np.load(self.y_paths[index]).astype(np.float64)

        return {"x": torch.as_tensor(self.x, dtype=torch.float32).unsqueeze(0),
                "y": torch.as_tensor(self.y, dtype=torch.float32).unsqueeze(0)}
       
    def __len__(self):  
        return self.len
        
train_dataset = dataset(train_X_dir, train_Y_dir)
test_dataset = dataset(test_X_dir, test_Y_dir) 


# Create training and test dataloaders

num_workers = 0
# how many samples per batch to load
batch_size = 10

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)



import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # adding NN
                # Linear function
        # input_dim = (16, 1)
        # hidden_dim = (4,1)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # # nonlinearity
        # self.sigmoid = nn.Sigmoid()
        # # linear function (readout)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x): #x = model(inputs)
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x #outputs

# initialize the NN
model = ConvAutoencoder()
print(model)


# specify loss function (MSE)
criterion = nn.MSELoss()
# specify loss function (SSIM)
# criterion = pytorch_ssim.SSIM(window_size=2)

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

iter = 0
n_epochs = 100
train_loss_l=np.zeros((n_epochs+1,1))
test_loss_l=np.zeros((n_epochs+1,1))
accuracy_l = np.zeros((n_epochs+1,1))
for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for iteration, batch in enumerate(train_loader):
        # print(list(enumerate(X_dataset)))
        # _ stands in for labels, here
        # no need to flatten images
        images = batch["x"]
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # calculate the loss
        loss = 1 - criterion(outputs, batch["y"])
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
        
        
        
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        train_loss_l[epoch]=train_loss
        print('Epoch: {} \tTraining Loss: {:.6f} \t '.format(
            epoch, 
            train_loss
            ))
        
# Plot the misfit MSE
train_loss_l = train_loss_l[1:,:]
fig576=plt.figure(figsize=(18,7))
plt.plot(train_loss_l,'-bo')
plt.xlabel('Iteration number')
plt.ylabel('Train Loss (MSE)')
plt.title(('Evolution of train loss'))
plt.grid(color='k', linestyle='--', linewidth=1)
ax=plt.gca()
ax.set_xlim(-0.5, n_epochs-1+0.5)
plt.xticks(np.linspace(0,n_epochs-1,n_epochs))    
np.save('C:/Users//train_loss_100.npy',train_loss_l)

# saving the model
# (1)
PATH ='C:/Users//model_3900.pth'
torch.save(model.state_dict(), PATH)  
# loading the model
# model = ConvAutoencoder(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()  

# (2)
PATH2 ='C:/Users//model_entire_3900.pth'
torch.save(model, PATH2)

# loading the model 
# model = torch.load(PATH)
# model.eval()
############################################### Test ###########################################################

# calculate accuracy
total_num = 0
total_loss = 0
criterion = nn.MSELoss()
# iterate throught test datasets

for it, im in enumerate(test_loader):

    imx = im["x"]
    imy = im["y"]

    outputs = model(imx)
    loss = criterion(outputs, batch["y"])
    if outputs.shape[0] == 1:

        image = outputs[0,0,:,:]
        image = image.detach().numpy()
        fig575= plt.figure(figsize=(12,12))
        plt.imshow(np.squeeze(image), cmap='gray',aspect='auto',extent=[0,2000,2000,0]) 
        mini = np.min(np.min(np.squeeze(image)))
        maxi = np.max(np.max(np.squeeze(image)))
        print('iter: {} \t min value: {} \t max value: {} '.format(
            it, 
            mini,
            maxi
            ))
        cbar = plt.colorbar(shrink=0.9)
        cbar.set_label('amplitudes(m/s)', rotation=90)
        plt.title('time section', fontsize=20)
        plt.xlabel('Distance [m]', fontsize=18)
        plt.ylabel('Depth [s]', fontsize=18) 
       
    else:
        for i in range(outputs.shape[0]):
            image = outputs[i,0,:,:]
            image = image.detach().numpy()
            fig576= plt.figure(figsize=(12,10))
            plt.imshow(np.squeeze(image), cmap='gray',aspect='auto',extent=[0,2000,2000,0]) 
            cbar = plt.colorbar(shrink=0.9)
            mini = np.min(np.min(np.squeeze(image)))
            maxi = np.max(np.max(np.squeeze(image)))
            print('iter: {} \t min value: {} \t max value: {} '.format(
                it, 
                mini,
                maxi))
            cbar.set_label('amplitudes(m/s)', rotation=90)
            plt.title('time section', fontsize=20)
            plt.xlabel('Distance [m]', fontsize=18)
            plt.ylabel('Depth [s]', fontsize=18)
      
        
    # print(it, outputs.shape)

    total_loss +=float(loss)
    total_num += 1

print(total_loss/total_num)

# ################################################### Cost Visualizations ################################################             
#     #     total_num = 0
#     #     total_loss = 0
#     #     criterion = nn.MSELoss()
        
            
#     ## iterate throught test datasets
# for it, im in enumerate(test_loader):
#     # print(it)
#     # print(im)
  
#     im_time = im["x"]
#     im_depth = im["y"]

#     outputs = model(im_time)
    
   
#     outputs = outputs.view(batch_size, 1, 480, 200)
#     outputs = outputs.detach().numpy()
    
#     # fig574= plt.figure(figsize=(12,18))
#     # plt.imshow(np.squeeze(im_time[it,:,:]), cmap='gray')
    
#     fig575= plt.figure(figsize=(12,7))
#     plt.imshow(np.squeeze(outputs[it,:,:]), cmap='gray')           
#     #     # total number of labels
#     #     total_loss +=float(loss)
#     #     total_num+=1
#     #     test_loss = total_loss/total_num
#     #     test_loss_l[epoch]=test_loss
#     #     # print(total_loss/total_num)    
#     #     print('total loss: {} \t '.format(test_loss))
   









         
  
            





# Plot the misfit MSE
test_loss_l = test_loss_l[1:,:]
fig577=plt.figure(figsize=(18,7))
plt.plot(test_loss_l,'-bo')
plt.xlabel('Iteration number')
plt.ylabel('Total Loss (MSE)')
plt.title(('Evolution of total loss'))
plt.grid(color='k', linestyle='--', linewidth=1)
ax=plt.gca()
ax.set_xlim(-0.5, n_epochs-1+0.5)
plt.xticks(np.linspace(0,n_epochs-1,n_epochs))



# See the true datasets











