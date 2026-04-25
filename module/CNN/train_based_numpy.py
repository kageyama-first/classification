from dataset import get_loader,set_seed
from module.CNN.CNN_module import CNN
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def train_epoch(model,data_loader,train=False,return_preds=False):
    total_loss=0
    all_preds=[]
    all_labels=[]
    correct=0
    total=0
    forward=model.forward
    backward=model.backward
    cross_entropy=model.cross_entropy
    for images,labels in data_loader:
        x=images.numpy().astype(np.float32)
        y=labels.numpy()
        preds=forward(x)
        pred_class=np.argmax(preds,axis=1)
        if train:
            backward(preds,y)
            loss=cross_entropy(preds,y)
            total_loss+=loss
            correct+=np.sum(pred_class==y)
            total+=len(y)
        else:
            all_preds.extend(pred_class)
            all_labels.extend(y)
    if train:
        acc_rate=correct/total
        avg_loss=total_loss/total
        return acc_rate,avg_loss
    else:
        acc=accuracy_score(all_labels,all_preds)
        f1=f1_score(all_labels,all_preds,average='macro')
        if return_preds:
            return acc,f1,all_preds,all_labels
        return acc,f1

def plot_confusion_matrix(labels,preds,class_names):
    cm=confusion_matrix(labels,preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__=='__main__':
    set_seed(42)
    class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    #实验
    strategies=['none', 'standard','weak','strong']
    epoches=10
    results={}
    for strategy in strategies:
        print(f"\n=== Loading data with strategy: {strategy} ===")
        train_loader, test_loader, val_loader = get_loader(strategy, batch_size=64)
        
        model = CNN(num_classes=6,lr=0.01)
        best_val_acc=0
        for epoch in range(epoches):
            train_acc,train_loss=train_epoch(model,train_loader,train=True)
            val_acc,val_f1=train_epoch(model,val_loader)
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        # 测试
        test_acc,test_f1,preds,labels=train_epoch(model, test_loader,return_preds=True)
        results[(strategy)] = (test_acc, test_f1, preds, labels)
        print(f"Test: acc={test_acc:.4f}, macro_f1={test_f1:.4f}")
        plot_confusion_matrix(labels, preds, class_names)
    
    print("\n=== Comparison Results ===")
    print("Strategy\tAccuracy\tMacro-F1")
    for strat, (acc, f1, _, _) in results.items():
        print(f"{strat}\t{acc:.4f}\t\t{f1:.4f}")