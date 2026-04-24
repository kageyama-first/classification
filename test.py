import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

from dataset import get_loader, set_seed

class SimpleCNN(nn.Module):
    """无 BatchNorm，无 Dropout，网络较浅"""
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 输入尺寸: 3 x 224 x 224
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x 112 x 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x 56 x 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 x 28 x 28
            nn.AdaptiveAvgPool2d((7, 7)) # 强制统一下一步尺寸为 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AdvancedCNN(nn.Module):
    """引入 BatchNorm、Dropout，增加网络深度和通道数"""
    def __init__(self, num_classes):
        super(AdvancedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #32 x 112 x 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #64 x 56 x 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #128 x 28 x 28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),  #256 x 14 x 14
            
            nn.AdaptiveAvgPool2d((4, 4)) #256 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

#训练和测试函数
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
        running_val_loss = 0.0
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
                running_val_loss+= loss.item()*inputs.size(0)
        
        #计算train指标
        avg_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(epoch_labels['train'], epoch_preds['train'])
        history_train_loss.append(avg_train_loss)
        history_train_acc.append(epoch_train_acc)
        #计算val指标
        epoch_val_f1=f1_score(epoch_labels['val'],epoch_preds['val'])
        
        #最佳模型
        if epoch_val_f1>best_val_f1:
            best_val_f1=epoch_val_f1
            best_epoch=epoch+1
            torch.save(model.state_dict(),"best_model.pth")
        
    return history_train_acc,history_train_loss,best_epoch,best_val_f1

#绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        ax=ax,
        xticks_rotation='vertical'
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

#绘制曲线
def plot_different_strategy_curves(results_dict,axis_y_name,title,save_path=None):
    
    plt.figure(figsize=(10, 6))
    
    # 定义策略的显示名称和颜色样式
    strategies_plot = {
        'none': {'label': 'Strategy:None', 'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'standard': {'label': 'Strategy:Standard', 'color': 'green', 'marker': 's', 'linestyle': '-'},
        'weak': {'label': 'Strategy:Weak', 'color': 'orange', 'marker': '^', 'linestyle': '-'},
        'strong': {'label': 'Strategy:Strong', 'color': 'red', 'marker': 'd', 'linestyle': '-'}
    }
    
    for strategy, losses in results_dict.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses,
                label=strategies_plot[strategy]['label'],
                color=strategies_plot[strategy]['color'],
                marker=strategies_plot[strategy]['marker'],
                linestyle=strategies_plot[strategy]['linestyle'],
                linewidth=2,
                markersize=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(axis_y_name, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

#绘制综合分组柱状图
def plot_metrics_grouped_bar(csv_data, model_name, save_path=None):
    df_model = csv_data[csv_data['Model'] == model_name]
    strategies = df_model['Strategy'].tolist()
    
    val_acc = df_model['Val_Acc'].tolist()
    val_f1 = df_model['Val_Macro_F1'].tolist()
    test_acc = df_model['Test_Acc'].tolist()
    test_f1 = df_model['Test_Macro_F1'].tolist()

    x = np.arange(len(strategies))
    width = 0.2  # 柱子宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, val_acc, width, label='Val Acc', color='#4c72b0')
    ax.bar(x - 0.5*width, test_acc, width, label='Test Acc', color='#55a868')
    ax.bar(x + 0.5*width, val_f1, width, label='Val F1', color='#c44e52')
    ax.bar(x + 1.5*width, test_f1, width, label='Test F1', color='#8172b3')

    ax.set_ylabel('Scores')
    ax.set_title(f'{model_name}: Val & Test Metrics Across Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

#绘制summary 表格
def summary(summary_results):
    df_summary = pd.DataFrame(summary_results)
    
    fig, ax = plt.subplots(figsize=(12, len(df_summary) * 0.5 + 1.5)) 
    ax.axis('off')
    ax.set_title("summary", fontsize=16, pad=10)

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        loc='center',
        cellLoc='center'
    )

    # 美化表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8) # 调整单元格宽度和高度的比例

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='black') # 字体加粗
            cell.set_facecolor('#d3d3d3') # 浅灰色背景
        elif col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e3e3e3')

    plt.tight_layout()
    summary_save_path = os.path.join("results", 'experiment_summary_table.png')
    plt.savefig(summary_save_path, dpi=300, bbox_inches='tight')
    print(f"Summary table image saved to: {summary_save_path}")
    plt.show()


if __name__ == '__main__':
    # 固定随机种子
    set_seed(42)
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    # Plot 1: none & standard
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, s in zip(axes, ['none', 'standard']):
        ConfusionMatrixDisplay.from_predictions(
            adv_test_results[s]['labels'], adv_test_results[s]['preds'],
            display_labels=class_names, cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical'
        )
        ax.set_title(f"AdvancedCNN - Strategy: {s}")
    plt.tight_layout()
    plt.savefig('adv_cm_none_standard.png', dpi=300)
    plt.show()

    # Plot 2: weak & strong
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, s in zip(axes, ['weak', 'strong']):
        ConfusionMatrixDisplay.from_predictions(
            adv_test_results[s]['labels'], adv_test_results[s]['preds'],
            display_labels=class_names, cmap=plt.cm.Oranges, ax=ax, xticks_rotation='vertical' # 换个颜色区分
        )
        ax.set_title(f"AdvancedCNN - Strategy: {s}")
    plt.tight_layout()
    plt.savefig('adv_cm_weak_strong.png', dpi=300)
    plt.show()
    
    #绘制summary表格
    summary(csv_results)

    # 同一CNN模型下不同策略中测试集和验证集的Acc/F1
    plot_metrics_grouped_bar(df_results, 'SimpleCNN', 'simplecnn_metrics_bar.png')
    plot_metrics_grouped_bar(df_results, 'AdvancedCNN', 'advancedcnn_metrics_bar.png')