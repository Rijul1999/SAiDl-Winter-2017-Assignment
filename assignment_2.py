import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#Read the training data and store it in y_train as a numpy array
train_df = pd.read_table("annotations/trainval.txt", header=None, sep='\s+')
y_train = train_df[0].as_matrix()

#Randomize the array
np.random.shuffle(y_train)

#Read the test data and store it in y_test as a numpy array
test_df = pd.read_table("annotations/test.txt", header=None, sep='\s+')
y_test = test_df[0].as_matrix()

#Randomize the array
np.random.shuffle(y_test)

#Store the classes and then convert the class data to numeric data
classes = ["Abyssinian", "american_bulldog","american_pit_bull_terrier","basset_hound","beagle","Bengal","Birman",
          "Bombay","boxer","British_Shorthair","chihuahua","Egyptian_Mau","english_cocker_spaniel","english_setter",
          "german_shorthaired","great_pyrenees","havanese","japanese_chin","keeshond","leonberger","Maine_Coon",
          "miniature_pinscher","newfoundland","Persian","pomeranian","pug","Ragdoll","Russian_Blue","saint_bernard",
          "samoyed","scottish_terrier","shiba_inu","Siamese","Sphynx","staffordshire_bull_terrier","wheaten_terrier",
          "yorkshire_terrier"]
ynp = np.zeros(3680)
for i in range(3680):
    t = y_train[i].rfind("_")
    st = y_train[i][0:t]
    ynp[i] = classes.index(st)
    
#Convert the numpy array to a torch Tensor
y = torch.from_numpy(ynp).type(torch.LongTensor)

#Define the convolutional netowrk
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3,96,11)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(96,32,3,stride=2, padding=4)
        self.conv3 = nn.Conv2d(32,16,3, padding=4)
        self.conv4 = nn.Conv2d(16,8,5)
        self.conv5 = nn.Conv2d(8,8,5)
        #Fully connected layers
        self.fc1 = nn.Linear(8*29*29,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,37)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = F.relu(self.conv3(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv4(x))
        print(x.shape)
        x = F.relu(self.conv5(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = x.view(-1,8*29*29)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Load the model into net
net = Net()

#Use Cross Entropic Loss and Adam optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999))

#Training the model
for epoch in range(2):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    size = 4 #Batch-size used is 4
    for i in range(int(3680/size)):
        
        #Read the images corresponding to the training data and store it as a numpy array
        for j in range(size):
            img_init = cv2.imread("images/"+str(y_train[i+j])+".jpg")
           
            x_init = cv2.resize(img_init,(500,500))
            if(j==0):
                xnp = x_init
            else:
                xnp = np.vstack((xnp,x_init))
    
        xnp = np.reshape(xnp,(size,3,500,500))
        
        #Convert the numpy array to a torch Tensor and then to a Variable
        x = torch.from_numpy(xnp).type(torch.FloatTensor)
        inputs,labels = Variable(x), Variable(y[i:i+size])
        
        #Feed forward and calculate the loss
        optimizer.zero_grad()
        print(labels.shape)
        outputs = net(inputs)
        print("outputs calculated")
        loss = criterion(outputs,labels)
        print("loss calculated")
        
        #Backpropagate and update weights
        loss.backward()
        print("backprop done")
        optimizer.step()
        print("Step taken")
          
        #Calculate and print the loss and accuracy
        running_loss = 0.0
        running_loss += loss.data[0]
        print("In epoch "+ str(epoch)+" the loss in step " + str(i)+ " is "+str(running_loss))
        _, predicted = torch.max(outputs.data,1)
        correct += (predicted == labels.data).sum()
        total += size
        accuracy = correct*100/total
        print("Correct: "+str(correct))
        print("total: "+str(total))
        print("Accuracy in epoch: "+str(epoch)+"and step: "+str(i)+" is "+str(accuracy))
        
print("Finished")

#Load the test data as a numpy array and convert it to a torch Tensor
ynp2 = np.zeros(3669)
for i in range(3669):
    t = y_test[i].rfind("_")
    st = y_test[i][0:t]
    ynp2[i] = classes.index(st)

y2 = torch.from_numpy(ynp2).type(torch.LongTensor)

#Feed forward into the neural network and calculate the accuracy over the test data
correct = 0.0
total = 0.0
size = 4
for i in range(int(3669/size)):
    for j in range(size):
        img_init2 = cv2.imread("images/"+str(y_test[i+j])+".jpg")
           
        x_init2 = cv2.resize(img_init2,(500,500))
        if(j==0):
            xnp2 = x_init2
        else:
            xnp2 = np.vstack((xnp2,x_init2))
    
    xnp2 = np.reshape(xnp2,(size,3,500,500))
        
    x2 = torch.from_numpy(xnp2).type(torch.FloatTensor)
    inputs,labels = Variable(x2), Variable(y2[i:i+size])
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += size
    correct += (predicted == labels.data).sum()

accuracy = correct*100/total
print("Accuracy on test set is "+ str(accuracy))
        
