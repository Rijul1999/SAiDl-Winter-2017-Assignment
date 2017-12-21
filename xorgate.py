import numpy as np

#Activation used is sigmoid
def sigmoid(a):
    return 1/(1+np.exp(-a))


def derive(a):
    return a*(1-a)

#Function to feed forward into the neural net
def forward(x,theta1,theta2,theta3,b_hid_1,b_hid_2,bout):
    a1 = np.dot(x,theta1) + b_hid_1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1,theta2) + b_hid_2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2,theta3) + bout
    h = sigmoid(a3)
    
    return np.round(h,decimals=0)


#Randomly initialize the inputs and stack them
x1 = np.random.randint(0,high=2,size=100)
x2 = np.random.randint(0,high=2,size=100)

y = np.bitwise_xor(x1,x2)
x = np.column_stack((x1,x2))

y = np.reshape(y,(100,1))

#Initialize the weights and biases for the input and two hidden layers
theta1 = np.random.rand(2,3)
theta2 = np.random.rand(3,3)
theta3 = np.random.rand(3,1)

bout = np.random.rand(1,1)
b_hid_2 = np.random.rand(1,3)
b_hid_1 = np.random.rand(1,3)

num_iter = 10000
lr = 0.01

#Train the model
for i in range(num_iter):
    
    #Forward propagation
    a1 = np.dot(x,theta1) + b_hid_1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1,theta2) + b_hid_2
    z2 = sigmoid(a2)
    
    a3 = np.dot(z2,theta3) + bout
    h = sigmoid(a3)
    
    #Backpropagation
    E_out = h-y
    delta_out = E_out*derive(h)
    
    E_hid_2 = np.dot(delta_out,theta3.T)
    delta_hid_2 = E_hid_2*derive(z2)
    
    E_hid_1 = np.dot(delta_hid_2,theta2.T)
    delta_hid_1 = E_hid_1*derive(z1)
    
    #Update weights and biases
    theta1 -= np.dot(x.T,delta_hid_1)*lr
    theta2 -= np.dot(z1.T,delta_hid_2)*lr
    theta3 -= np.dot(z2.T,delta_out)*lr
    
    b_hid_1 -= np.sum(delta_hid_1,axis=0)*lr
    b_hid_2 -=np.sum(delta_hid_2,axis=0)*lr
    bout -=np.sum(delta_out,axis=0)*lr
    
#Randomly initialize the test data   
testx1 = np.random.randint(0,high=2,size=10)
testx2 = np.random.randint(0,high=2,size=10)
testx3 = np.random.randint(0,high=2,size=10)
testx4 = np.random.randint(0,high=2,size=10)

testX1 = np.column_stack((testx1,testx2))
testX2 = np.column_stack((testx3,testx4))

#Calculate the output to the test data
Y1 = forward(testX1,theta1,theta2,theta3,b_hid_1,b_hid_2,bout)
Y2 = forward(testX2,theta1,theta2,theta3,b_hid_1,b_hid_2,bout)

#Print the results
for i in range(10):
    s1 = str(testX1[i][0])+""+str(testX2[i][0])
    s2 = str(testX1[i][1])+""+str(testX2[i][1])
    out = str(int(Y1[i])) + "" +str(int(Y2[i]))
    print("The bitwise XOR of "+s1+" and "+s2+" is "+out)
