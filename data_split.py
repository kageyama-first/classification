import os
import random
from shutil import copy2

def data_split(src,dst,train_scale=0.7,val_scale=0.15,test_scale=0.15):
    categories=[cat for cat in os.listdir(src)]
    
    #创建文件夹
    split_names=['train','val','test']
    for s in split_names:
        split_path=os.path.join(dst,s)
        os.makedirs(split_path,exist_ok=True)
        for cat in categories:
            cat_split_path=os.path.join(split_path,cat)
            os.makedirs(cat_split_path,exist_ok=True)
    #获取图片
    for cat in categories:
        cur_data_path=os.path.join(src,cat)
        cur_data=[f for f in os.listdir(cur_data_path) if f.lower().endswith(('.jpg','.png' , '.jpeg'))]
        cur_data_len=len(cur_data)
        cur_data_list=list(range(cur_data_len))
        random.shuffle(cur_data_list)
        
        #划分图片
        train_folder=os.path.join(dst,'train',cat)
        val_folder=os.path.join(dst,'val',cat)
        test_folder=os.path.join(dst,'test',cat)
        
        train_stop=int(cur_data_len*train_scale)
        val_stop=int(cur_data_len*(val_scale+train_scale))
        
        for idx, file_idx in enumerate(cur_data_list):
            src_path=os.path.join(cur_data_path,cur_data[file_idx])
            if idx<train_stop:
                copy2(src_path,train_folder)
            elif idx<val_stop:
                copy2(src_path,val_folder)
            else:
                copy2(src_path,test_folder)
    print('划分完成')

if __name__ =="__main__":
    src = "C:\\Users\\17519\\Desktop\\classification\\data"
    dst = "C:\\Users\\17519\\Desktop\\classification\\data_split"
    data_split(src,dst)
