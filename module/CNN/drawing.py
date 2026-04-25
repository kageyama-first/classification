import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import ConfusionMatrixDisplay

#绘制混淆矩阵
def plot_Confusion_Matrix(adv_test_results,class_names,title,plot_save_path=None):
        # Plot 1: none & standard
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, s in zip(axes, ['none', 'standard']):
            ConfusionMatrixDisplay.from_predictions(
                adv_test_results[s]['labels'], adv_test_results[s]['preds'],
                display_labels=class_names, cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical'
            )
            ax.set_title(f"{title}: {s}")
        plt.tight_layout()
        plt.savefig(save_path=plot_save_path, dpi=300)
        plt.show()

        # Plot 2: weak & strong
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, s in zip(axes, ['weak', 'strong']):
            ConfusionMatrixDisplay.from_predictions(
                adv_test_results[s]['labels'], adv_test_results[s]['preds'],
                display_labels=class_names, cmap=plt.cm.Oranges, ax=ax, xticks_rotation='vertical' # 换个颜色区分
            )
            ax.set_title(f"{title}: {s}")
        plt.tight_layout()
        plt.savefig(save_path=plot_save_path, dpi=300)
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
    ax.bar(x - 1.5*width, val_acc, width, label='Val Acc', color="#9DBEF1")
    ax.bar(x - 0.5*width, test_acc, width, label='Test Acc', color="#5dba73")
    ax.bar(x + 0.5*width, val_f1, width, label='Val F1', color="#f08d90")
    ax.bar(x + 1.5*width, test_f1, width, label='Test F1', color="#aca1d1")

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