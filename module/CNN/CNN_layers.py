import numpy as np

#卷积层
def conv(input_data,kernels,stride=1,padding=0):
    #填充
    if padding > 0:
        pad_width=((0,0),(0,0),(padding,padding),(padding,padding))
        input_data=np.pad(input_data,pad_width,mode='constant')
    
    #输入输出维度
    batch_size,channel_in,input_h,input_w = input_data.shape
    channel_out,channel_in_k,kernel_h,kernel_w = kernels.shape
    assert channel_in==channel_in_k
    
    out_h = (input_h-kernel_h)//stride+1
    out_w = (input_w-kernel_w)//stride+1
    
    output=np.zeros((batch_size,channel_out,out_h,out_w))
    for b in range(batch_size):
        for c in range(channel_out):
            kernel=kernels[c]
            for i in range(out_h):
                for j in range(out_w):
                    region=input_data[b,:,i*stride : i*stride+kernel_h,j*stride : j*stride+kernel_w]
                    output[b,c,i,j]=np.sum(region*kernel)
    return output

#ReLU 激活
def relu(x):
    return np.maximum(0,x)

#最大池化
def max_pool(input_data,pool_size=2,stride=2):
    #输入输出维度
    batch_size,channel_in,input_h,input_w= input_data.shape
    out_h=(input_h-pool_size)//stride +1
    out_w=(input_w-pool_size)//stride +1
    output=np.zeros((batch_size,channel_in,out_h,out_w))
    
    max_pos={}
    for b in range(batch_size):
        for c in range(channel_in):
            for i in range(out_h):
                for j  in range(out_w):
                    region=input_data[b,c,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
                    output[b,c,i,j]=np.max(region)
                    #记录最大值的位置
                    idx=np.argmax(region)#扁平化后最大值的索引
                    r=idx//pool_size
                    c_offset=idx%pool_size
                    max_pos[(b,c,i,j)]=(r,c_offset)
    return output,max_pos

#展平
def flatten(input_data):
    B,C,H,W=input_data.shape
    return input_data.reshape(B,-1)

#全连接层
class  Dense:
    def __init__(self,input_size,output_size):
        self.w =np.random.randn(output_size,input_size)*0.01
        self.b =np.zeros((output_size,1))
    
    def forward(self,x):
        self.x=x #x=(B,D_in)
        output=np.dot(self.w,x.T)+self.b #(D_out,D_in)*(D_in,B)
        return output.T
    
    def backward(self,d_out,lr=0.01):
        d_out=d_out.T
        dw=np.dot(d_out,self.x.T) #矩阵对应元素相乘，.T表示矩阵的转置
        db=np.sum(d_out,axis=1,keepdims=True)
        dx=np.dot(self.w.T,d_out) #(D_in,B)
        
        self.w-=lr*dw
        self.b-=lr*db
        return dx.T
        

#Softmax 输出
def softmax(x):
    exp_x=np.exp(x-np.max(x,axis=1,keepdims=True)) #i->e^i ,减去最大值，防止溢出，并且该函数有平移不变性
    return exp_x/np.sum(exp_x,axis=1,keepdims=True) #归一化

#交叉熵损失
def cross_entropy(pred,label):
    #pred:(B,num_classes)
    #label(B,) 真实类别索引
    #返回平均标量损失
    B=pred.shape[0]
    loss=0.0
    for i in range(B):
        loss+= -np.log(pred[i,label[i]]+1e-9) #取自然对数;1e-9极小值，防止为0
    return loss/B

#反向传播
def softmax_cross_entropy_backward(pred,label):
    B,C=pred.shape
    grad=pred.copy() #模型对B个图片预测的每类的概率，（i，j）是对第i张图片预测的为第j类的概率
    for i in range(B):
        grad[i,label[i]]-=1
    return grad/B

def relu_back(d_out,x):
    dx=d_out.copy()
    dx[x<=0]=0
    return dx

def max_pool_back(d_out,max_pos,pool_size=2,stride=2):
    #d_out: 损失对池化输出的梯度 (B, C, H_out, W_out)
    batch_size,channel_out,out_h,out_w=d_out.shape
    input_h=(out_h-1)*stride+pool_size
    input_w=(out_w - 1) * stride + pool_size
    dx=np.zeros((batch_size,channel_out,input_h,input_w))
    
    for b in range(batch_size):
        for c in range(channel_out):
            for i in range(out_h):
                for j in range(out_w):
                    r,c_off=max_pos[(b,c,i,j)]
                    dx[b,c,i*stride+r,j*stride+c_off]+=d_out[b,c,i,j]#还原时，记录的最大值位置处为梯度值，其余为零。
    return dx

def conv_backward(input_data,kernels,d_out,stride=1,padding=0):
    #先恢复原始输入大小
    if padding > 0:
        pad_width=((0,0),(0,0),(padding,padding),(padding,padding))
        input_data=np.pad(input_data,pad_width,mode='constant')
    
    #尺寸
    B,C_in,input_h,input_w=input_data.shape
    C_out,C_in_k,kernel_h,kernel_w=kernels.shape
    assert C_in==C_in_k
    _,_,out_h,out_w=d_out.shape
    
    dx=np.zeros_like(input_data)
    dk=np.zeros_like(kernels)
    
    #计算梯度
    for b in range(B):
        for c in range(C_out):
            for i in range(out_h):
                for j in range(out_w):
                    grad=d_out[b,c,i,j]
                    #对输入的梯度，叠加到输入窗口
                    dx[b,:,i*stride:i*stride+kernel_h,j*stride:j*stride+kernel_w]+=grad*kernels[c]
                    #对卷积的梯度，累加
                    input_region=input_data[b,:,i*stride:i*stride+kernel_h,j*stride:j*stride+kernel_w]
                    dk[c]+=grad*input_region
    
    #去掉填充
    if padding>0:
        dx=dx[:,:,padding:input_h-padding,padding:input_w-padding]
    
    return dx,dk


#正则化
class dropout:
    def __init__(self,p=0.5):
        self.p=p
        self.mask=None
        
    def forward(self,x,train=True):
        if train:
            self.mask=(np.random.rand(*x.shape)>self.p)/(1-self.p)
            return x*self.mask
        else:
            return x
    
    def backward(self,d_out):
        return d_out*self.mask