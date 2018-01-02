import h5py
import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms 
import numpy as np

# Encoder network to convert the input image into a feature map
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,3,padding=1),   # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)   # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),  # batch x 256 x 7 x 7
                        nn.ReLU()
        )
        
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(batch_size, -1)
        return out
    
encoder = Encoder()


#Decoder network to convert the feature map back into the input image
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1),  # batch x 128 x 14 x 14
                        nn.ReLU(),                            
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,128,3,1,1),    # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),     # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,64,3,1,1),      # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),      # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),      # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),     # batch x 1 x 28 x 28
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(batch_size,256,7,7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out
    
decoder = Decoder()


batch_size = 100

#Load the MNIST dataset via the DataLoader function of PyTorch
mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)

#Convert the torch tensors to numpy arrays and stack them
c = 0
for images, labels in train_loader:
    if(c == 0):
        img_np_hdf_train = images.numpy()
        
    else:
        img_np_hdf_train = np.vstack((img_np_hdf_train, images.numpy()))
        
    c+=1



c = 0
for im, la in test_loader:
    if(c == 0):
        img_np_hdf_test = im.numpy()
        
    else:
        img_np_hdf_test = np.vstack((img_np_hdf_test, im.numpy()))
        
    c+=1


# The random noise element of the image
noise = torch.rand(batch_size, 1, 28, 28)

#Store the numpy arrays as hdf5 datasets in a hdf5 file named mnistdataset
f = h5py.File("mnistdataset.hdf5", "w")
trainset_img = f.create_dataset("train_images", data=img_np_hdf_train)
testset_img = f.create_dataset("test_images", data=img_np_hdf_test)


#Set the hyperparameters
epoch = 10
learning_rate = 0.0005

#Use the mean squared loss and Adam optimizer with default betas
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(encoder.parameters())+ list(decoder.parameters()), lr = learning_rate)


#Train the model by sending noisy input into the encoder and taking loss against the original image in the decoder
for i in range(epoch):
    for image,label in train_loader:
        image_n = torch.mul(image, noise)
        image = Variable(image)
        image_n = Variable(image_n)
        
        optimizer.zero_grad()
        output = encoder(image_n)
        output = decoder(output)
        loss = criterion(output,image)
        loss.backward()
        optimizer.step()
        print("Training loss in epoch "+str(i)+ " is:")
        print(loss.data)


#Find the loss in the test dataset
for timage,tlabel in test_loader:
    test_img_n = torch.mul(timage, noise)
    timage = Variable(timage)
    test_img_n = Variable(test_img_n)
    
    output = encoder(test_img_n)
    output = decoder(output)
    loss = criterion(output, timage)
    print("Test loss is:")
    print(loss.data)




