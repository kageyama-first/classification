from module.CNN.CNN_layers import *
import numpy as np


class CNN:
    def __init__(self,num_classes=6,lr=0.01):
        self.lr=lr
        self.conv1_kernels=np.random.randn(16,3,3,3).astype(np.float32)*0.01
        self.conv2_kernels=np.random.randn(32,16,3,3).astype(np.float32)*0.01
        self.conv3_kernels=np.random.randn(64,32,3,3).astype(np.float32)*0.01
        self.dense1=Dense(64,32)
        self.dense2=Dense(32,num_classes)
        self.cache={}
        self.pool_size=2
        pass
    
    def forward(self,input_data):
        self.cache={}
        #第一层卷积(3,224,224)-(16,112,112)
        self.cache['input_data']=input_data
        out=conv(input_data,self.conv1_kernels,stride=1,padding=1)
        self.cache['conv1_out']=out
        out=relu(out)
        out,self.cache['pos1']=max_pool(out,pool_size=2,stride=2)
        self.cache['max_pool1_out']=out
        
        #第二层卷积(16,112,112)-(32,56,56)
        out=conv(out,self.conv2_kernels,stride=1,padding=1)
        self.cache['conv2_out']=out
        out=relu(out)
        out,self.cache['pos2']=max_pool(out,pool_size=self.pool_size,stride=2)
        self.cache['max_pool2_out']=out
        
        #第三层卷积(32,56,56)-(64,28,28)
        out=conv(out,self.conv3_kernels,stride=1,padding=1)
        self.cache['conv3_out']=out
        out=relu(out)
        out,self.cache['pos3']=max_pool(out,pool_size=self.pool_size,stride=2)
        self.cache['max_pool3_out']=out
        
        #全局平均池化
        gap_out=np.mean(out,axis=(2,3),keepdims=False)
        
        #dense1
        out=self.dense1.forward(gap_out)
        out=relu(out)
        self.cache['relu_dense1_out']=out
        #dense2
        out=self.dense2.forward(out)
        out=softmax(out)
        return out
    
    def cross_entropy(self,preds,labels):
        return cross_entropy(preds,labels)
    
    def backward(self,preds,labels):
        #全连接层反向
        grad=softmax_cross_entropy_backward(preds,labels)
        grad=self.dense2.backward(grad,self.lr)
        grad[self.cache['relu_dense1_out']<=0]=0
        grad=self.dense1.backward(grad,self.lr)
        
        #全局平均池化反向
        B, C = grad.shape
        H, W = self.cache['max_pool3_out'].shape[2:]
        grad = grad[:, :, np.newaxis, np.newaxis]# (B, C, 1, 1)
        grad = np.broadcast_to(grad, (B, C, H, W))# (B, C, H, W)
        grad = grad / (H * W)
        
        # 池化3反向
        grad = max_pool_back(grad, self.cache['pos3'], pool_size=self.pool_size, stride=2)
        grad[self.cache['conv3_out']<=0]=0
        grad, dk3 = conv_backward(self.cache['max_pool2_out'], self.conv3_kernels, grad, stride=1, padding=1)
        self.conv3_kernels -= self.lr * dk3
        
        # 池化2反向
        grad = max_pool_back(grad, self.cache['pos2'], pool_size=self.pool_size, stride=2)
        grad[self.cache['conv2_out']<=0]=0
        grad, dk2 = conv_backward(self.cache['max_pool1_out'], self.conv2_kernels, grad, stride=1, padding=1)
        self.conv2_kernels -= self.lr * dk2
        
        # 池化1反向
        grad = max_pool_back(grad, self.cache['pos1'], pool_size=2, stride=2)
        grad[self.cache['conv1_out']<=0]=0
        grad, dk1 = conv_backward(self.cache['input_data'], self.conv1_kernels, grad, stride=1, padding=1)
        self.conv1_kernels -= self.lr * dk1
        
        return self


