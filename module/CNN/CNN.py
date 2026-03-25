import numpy as np

#卷积层
def conv2d(input_data,kernel,stride=1,padding=0):
    if padding > 0:
        input_data=np.pad(input_data,((padding,padding),(padding,padding)),mode='constant')
    input_h,input_w = input_data.shape
    kernel_h,kernel_w = kernel.shape
    
    out_h = (input_h-kernel_h)//stride+1
    out_w = (input_w-kernel_w)//stride+1
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            region=input_data[i*stride : i*stride+kernel_h,j*stride : j*stride+kernel_w]
            output[i,j]=np.sum(region*kernel)
    return output

#ReLU 激活
def relu(x):
    return np.maximum(0,x)

#最大池化
def max_pool2d(input_data,pool_size=2,stride=2):
    input_h,input_w= input_data.shape
    out_h=(input_h-pool_size)//stride +1
    out_w=(input_w-pool_size)//stride +1
    
    output=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j  in range(out_w):
            region=input_data[i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
            output[i,j]=np.max(region)
            
            
    return output

#全连接层
class  Dense:
    def __init__(self,input_size,output_size):
        self.w =np.random.randn(output_size,input_size)*0.01
        self.b =np.zeros((output_size,1))
    def forward(self,x):
        self.x=x
        return np.dot(self.w,x)+self.b

#Softmax 输出
def softmax(x):
    exp_x=np.exp(x-np.max(x)) #i->e^i
    return exp_x/np.sum(exp_x) #归一化

#交叉熵损失
def cross_entropy(pred,label):
    return -np.log(pred[label]+1e-9) #取自然对数;1e-9极小值，防止为0

#向前传播
input_image=np.random.randn(5,5)
kernel=np.random.randn(3,3)

conv_out=conv2d(input_image,kernel)
relu_out=relu(conv_out)
pool_out=max_pool2d(relu_out)
flatten=pool_out.flatten().reshape(-1,1) #(行，列)-1 表示行数不确定，自动计算

dense=Dense(flatten.shape[0],10) #flatten.shape[0],flatten行数
logits=dense.forward(flatten)
probs=softmax(logits)

print("输入图像：\n", input_image)
print("卷积输出：\n", conv_out)
print("ReLU输出：\n", relu_out)
print("池化输出：\n", pool_out)
print("分类概率：\n", probs)

#反向传播
