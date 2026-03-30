import pandas as pd
import matplotlib.pyplot as plt
import os

file = 'train_history_resnet101.csv'

dfs = []
if os.path.exists(file):
    dfs.append(pd.read_csv(file))

if not dfs:
    print("找不到檔案")
else:
    full_df = pd.concat(dfs).drop_duplicates(subset=['epoch']).sort_values('epoch')
    
    plt.figure(figsize=(15, 6))
    plt.style.use('seaborn-v0_8') 

    # Accuracy曲線
    plt.subplot(1, 2, 1)
    plt.plot(full_df['epoch'], full_df['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    plt.plot(full_df['epoch'], full_df['val_acc'], label='Val Accuracy', marker='s', markersize=3)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Loss曲線
    plt.subplot(1, 2, 2)
    plt.plot(full_df['epoch'], full_df['train_loss'], label='Train Loss', marker='o', markersize=3, color='indianred')
    plt.plot(full_df['epoch'], full_df['val_loss'], label='Val Loss', marker='s', markersize=3, color='steelblue')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    
    plt.savefig('training_curves_resnet101.png', dpi=300)
    print("training_curves_resnet101.png")
    
    # 顯示最後幾輪的數值
    print("\n最後 5 輪數據摘要：")
    print(full_df.tail())