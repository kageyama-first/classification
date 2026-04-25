import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from CNN_model_torchbased import *
from drawing import *

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from dataset import get_loader, set_seed


def test(model,loader,device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss=0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #计算损失
            loss=criterion(outputs,labels)
            total_loss += loss.item() * inputs.size(0)
            #正确率
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_loss = total_loss / len(loader.dataset)
    return acc, macro_f1,avg_loss,all_preds,all_labels


def train_and_evaluate(model,train_loader,val_loader,device, epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 记录训练过程中的损失和指标
    history_train_loss=[]
    history_train_acc=[]
    best_epoch=0
    best_val_f1=-1.0
    
    for epoch in range(epochs):
        #记录数据
        epoch_preds = {'train':[],'val':[]}
        epoch_labels = {'train':[],'val':[]}
        running_train_loss = 0.0
        #训练
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #记录数据
            _, preds = torch.max(outputs, 1)
            epoch_preds['train'].extend(preds.cpu().numpy())
            epoch_labels['train'].extend(labels.cpu().numpy())
            running_train_loss+= loss.item()* inputs.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        if  (epoch+1)%5==0 :
        #验证
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    
                    #记录数据
                    _, preds = torch.max(outputs, 1)
                    epoch_preds['val'].extend(preds.cpu().numpy())
                    epoch_labels['val'].extend(labels.cpu().numpy())
        
        #计算train指标
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(epoch_labels['train'], epoch_preds['train'])
        history_train_loss.append(avg_train_loss)
        history_train_acc.append(epoch_train_acc)
        #计算val指标
        epoch_val_f1=f1_score(epoch_labels['val'],epoch_preds['val'],average='macro')
        
        #最佳模型
        if epoch_val_f1>best_val_f1:
            best_val_f1=epoch_val_f1
            best_epoch=epoch+1
            torch.save(model.state_dict(),"best_model.pth")
        
    return history_train_acc,history_train_loss,best_epoch,best_val_f1


if __name__ == '__main__':
    # 固定随机种子
    set_seed(42)
    EPOCHS = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #存储实验结果
    if not os.path.exists("results"):
        os.makedirs("results")
    
    #动态获取类别
    tmp_train, _, _ = get_loader('none', batch_size=2, num_workers=1)
    class_names = tmp_train.dataset.classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")
    
    # 1.基础模型
    strategy=['none','standard','weak','strong']
    models_to_test = {
        'SimpleCNN': SimpleCNN,
        'AdvancedCNN': AdvancedCNN
    }
    
    # 用于记录作图与CSV数据
    metrics_history = {'train_loss': {}, 'train_acc': {}}
    csv_results = []
    
    # 保存 AdvancedCNN 的测试集预测结果
    adv_test_results = {}
    for model_name, ModelClass in models_to_test.items():
        metrics_history['train_loss'][model_name] = {}
        metrics_history['train_acc'][model_name] = {}
        
        for i, s in enumerate(strategy):
            print(f"\n[Experiment {i+1}] {model_name} + Strategy: {s}")
            model = ModelClass(num_classes)
            
            train_loader, test_loader, val_loader = get_loader(s, batch_size=32)
            
            # 训练与验证
            train_acc ,train_loss,best_epoch,best_val_f1= train_and_evaluate(model, train_loader, val_loader,device, epochs=EPOCHS)
            #最佳模型
            model.load_state_dict(torch.load("best_model.pth"))
            # 最终验证与测试
            val_acc,val_f1,val_loss,val_preds,val_labels=test(model, val_loader, device)
            test_acc, test_f1, test_loss, test_preds, test_labels = test(model, test_loader, device)
            
            # 存储曲线绘制数据
            metrics_history['train_loss'][model_name][s] = train_loss
            metrics_history['train_acc'][model_name][s] = train_acc
            
            # 存储CSV记录
            csv_results.append({
                'Model': model_name, 'Strategy': s,
                'Best_epoch': best_epoch,
                'Best_val_macro_f1':best_val_f1,
                'Val_Loss': f"{val_loss:.8f}", 'Val_Acc': round(val_acc,8), 'Val_Macro_F1': round(val_f1,8),
                'Test_Loss': f"{test_loss:.8f}", 'Test_Acc': round(test_acc,8), 'Test_Macro_F1': round(test_f1,8)
            })
            
            # 保存AdvancedCNN预测结果用于画混淆矩阵
            if model_name == 'AdvancedCNN':
                adv_test_results[s] = {'preds': test_preds, 'labels': test_labels}

    # 输出统计数据为 CSV
    df_results = pd.DataFrame(csv_results)
    df_results.to_csv("experiment_results.csv", index=False)
    print("\n" + "="*50)
    print("CSV file 'experiment_results.csv' successfully saved!")
    print(df_results)
    print("="*50 + "\n")

    # 绘制不同策略下的训练损失图
    plot_different_strategy_curves(
        metrics_history['train_loss']['SimpleCNN'],
        'Train Loss', 'SimpleCNN: Training Loss over Strategies',save_path=os.path.join("results", 'simplecnn_loss.png'))
    
    plot_different_strategy_curves(
        metrics_history['train_loss']['AdvancedCNN'],
        'Train Loss', 'AdvancedCNN: Training Loss over Strategies',save_path=os.path.join("results", 'advancedcnn_loss.png'))

    # 绘制不同策略下的训练正确率图
    plot_different_strategy_curves(
        metrics_history['train_acc']['SimpleCNN'],
        'Train Accuracy', 'SimpleCNN: Training Accuracy over Strategies', save_path=os.path.join("results", 'simplecnn_acc.png'))
    
    plot_different_strategy_curves(
        metrics_history['train_acc']['AdvancedCNN'],
        'Train Accuracy', 'AdvancedCNN: Training Accuracy over Strategies', save_path=os.path.join("results", 'advancedcnn_acc.png'))

    # 绘制 AdvancedCNN 在不同策略下的混淆矩阵
    plot_Confusion_Matrix(adv_test_results,class_names,title="AdvancedCNN - Strategy",plot_save_path=os.path.join("results",'adv_cnn_none_standard.png'))
    
    #绘制summary表格
    summary_for_plot = []
    for row in csv_results:
        summary_for_plot.append({
            'tag': f"{row['Model'].lower()}_{row['Strategy']}",
            'init': row['Model'].lower(),
            'augmentation': row['Strategy'],
            'best_epoch': row['Best_epoch'],
            'best_val_macro_f1': row['Best_val_macro_f1'],
            'test_acc': row['Test_Acc'],
            'test_macro_f1': row['Test_Macro_F1']
        })
    summary(summary_for_plot)

    # 同一CNN模型下不同策略中测试集和验证集的Acc/F1
    plot_metrics_grouped_bar(df_results, 'SimpleCNN', save_path=os.path.join("results",'simplecnn_metrics_bar.png'))
    plot_metrics_grouped_bar(df_results, 'AdvancedCNN', save_path=os.path.join("results",'advancedcnn_metrics_bar.png'))