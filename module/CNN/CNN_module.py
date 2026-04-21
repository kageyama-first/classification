from module.CNN.CNN_layers import *
import numpy as np


class CNN:
    def __init__(self,num_classes=6,lr=0.01):
        self.lr=lr
        self.conv1_kernels=np.random.randn(16,3,3,3)*0.01
        self.conv2_kernels=np.random.randn(32,3,3,3)*0.01
        self.conv3_kernels=np.random.randn(64,3,3,3)*0.01
        self.dense1=Dense(64*28*28,128)
        self.dense2=Dense(128,num_classes)
        self.cache={}
        self.pool_size=2
        self.max_pos={}
        pass
    
    def forward(self,input_data):
        self.cache={}
        #第一层卷积(3,224,224)-(1,112,112)
        self.cache['input_data']=input_data
        out=conv(input_data,self.conv1_kernels,stride=1,padding=1)
        self.cache['conv1_out']=out
        out=relu(out)
        self.cache['relu1_out']=out
        out,self.max_pos['pos1']=max_pool(out,pool_size=2,stride=2)
        self.cache['max_pool1_out']=out
        
        #第二层卷积(1,112,112)-(1,56,56)
        out=conv(out,self.conv2_kernels,stride=1,padding=1)
        self.cache['conv2_out']=out
        out=relu(out)
        self.cache['relu2_out']=out
        out,self.max_pos['pos2']=max_pool(out,pool_size=self.pool_size,stride=2)
        self.cache['max_pool2_out']=out
        
        #第三层卷积(1,56,56)-(1,28,28)
        out=conv(out,self.conv1_kernels,stride=1,padding=1)
        self.cache['conv3_out']=out
        out=relu(out)
        self.cache['relu3_out']=out
        out,self.max_pos['pos3']=max_pool(out,pool_size=self.pool_size,stride=2)
        self.cache['max_pool3_out']=out
        
        out=flatten(out)
        self.cache['flatten_out']=out
        
        #dense1
        out=self.dense1.forward(out)
        self.cache['dense1_out']=out
        out=relu(out)
        self.cache['relu_dense1_out']=out
        #dense2
        out=self.dense2.forward(out)
        out=softmax(out)
        self.cache['pred']=out
        return out
    
    def cross_entropy(self,labels):
        return cross_entropy(self.cache['pred'],labels)
    
    def backward(self,labels):
        #全连接层反向
        grad=softmax_cross_entropy_backward(self.pred,labels)
        grad=self.dense2.backward(grad,self.lr)
        grad=relu_back(self.cache['relu_dense1_out'])
        grad=self.dense1.backward(grad,self.lr)
        
        # 池化3反向
        grad = grad.reshape(self.cache['max_pool3_out'].shape)
        grad = max_pool_back(grad, self.max_pos['pos3'], pool_size=self.pool_size, stride=2)
        grad = relu_back(grad, self.cache['conv3_out'])
        grad, dk3 = conv_backward(self.cache['max_pool2_out'], self.conv3_kernels, grad, stride=1, padding=1)
        self.conv3_kernels -= self.lr * dk3
        
        # 池化2反向
        grad = max_pool_back(grad, self.max_pos['pos2'], pool_size=self.pool_size, stride=2)
        grad = relu_back(grad, self.cache['conv2_out'])
        grad, dk2 = conv_backward(self.cache['max_pool1_out'], self.conv2_kernels, grad, stride=1, padding=1)
        self.conv2_kernels -= self.lr * dk2
        
        # 池化1反向
        grad = max_pool_back(grad, self.max_pos['pos1'], pool_size=2, stride=2)
        grad = relu_back(grad, self.cache['conv1_out'])
        grad, dk1 = conv_backward(self.cache['input_data'], self.conv1_kernels, grad, stride=1, padding=1)
        self.conv1_kernels -= self.lr * dk1
        
        return self


