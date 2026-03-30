# ResNeXt-101
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import glob
import warnings
import gc

# 環境
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()

# 參數、路徑
DATA_DIR = '/kaggle/input/datasets/liupeilinlily/computer-vision-hw1/data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

NUM_CLASSES = 100
BATCH_SIZE = 4           
ACCUMULATION_STEPS = 16  
EPOCHS = 60
LEARNING_RATE = 5e-5 

# 資料處理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class FlatTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), os.path.basename(img_path)

def get_loaders():
    # 確保類別索引正確 (0-99) ->調整順序
    my_class_to_idx = {str(i): i for i in range(NUM_CLASSES)}
    def fix_ds(ds):
        ds.class_to_idx = my_class_to_idx
        ds.samples = [(p, my_class_to_idx[os.path.basename(os.path.dirname(p))]) for p, _ in ds.samples]
        return ds
    
    train_ds = fix_ds(datasets.ImageFolder(TRAIN_DIR, data_transforms['train']))
    val_ds = fix_ds(datasets.ImageFolder(VAL_DIR, data_transforms['val']))
    
    return {
        'train': DataLoader(train_ds, batch_size=BATCH_SIZE * (gpu_count if gpu_count > 1 else 1), 
                            shuffle=True, num_workers=2, pin_memory=False),
        'val': DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }

# 模型
def initialize_model():
    # 用預訓練權重做基礎
    model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
    # 輸出層符合100類，並加入Dropout防止過擬合
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    return model.to(device)

# 主程式
def train_model():
    loaders = get_loaders()
    model = initialize_model()
    if gpu_count > 1: model = nn.DataParallel(model)

    # 優化器、損失函數
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label Smoothing 提升泛化力
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    scaler = GradScaler() # 混合精度
    best_acc = 0.0
    history = []

    for epoch in range(EPOCHS):
        epoch_log = {'epoch': epoch + 1}
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            pbar = tqdm(loaders[phase], desc=f'Epoch {epoch+1}/{EPOCHS} [{phase}]')
            
            if phase == 'train':
                optimizer.zero_grad()
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss_accum = loss / ACCUMULATION_STEPS 
                    
                    scaler.scale(loss_accum).backward()

                    if (i + 1) % ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            else:
                with torch.no_grad():
                    for inputs, labels in pbar:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)
            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_log[f'{phase}_acc'] = epoch_acc.item()
            epoch_log[f'{phase}_loss'] = epoch_loss
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 存檔時考量 DataParallel
                state = model.module.state_dict() if gpu_count > 1 else model.state_dict()
                torch.save(state, 'best_resnext.pth')
                print(f"目前發現的最佳模型: {best_acc:.4f}")

        history.append(epoch_log)
        pd.DataFrame(history).to_csv('train_history.csv', index=False)
        scheduler.step()
        
        # 每輪結束後強制清理記憶體防止Kernel Died  ->但kaggle一定程度之後還是會死掉
        gc.collect()
        torch.cuda.empty_cache()

    return model

# Ten-Crop TTA
def run_inference(model):
    print("\nTen-Crop TTA 推論")
    # 載入訓練過程中的最佳權重
    if os.path.exists('best_resnext.pth'):
        state = torch.load('best_resnext.pth')
        if gpu_count > 1: model.module.load_state_dict(state)
        else: model.load_state_dict(state)
    
    model.eval()
    tta_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(c) for c in crops])),
    ])
    
    test_ds = FlatTestDataset(TEST_DIR, transform=tta_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    results = []
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc='Predicting'):
            bs, n_crops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w).to(device)
            with autocast(device_type='cuda'):
                outputs = model(inputs)
            # 平均10個裁切區域的Softmax機率
            avg_probs = torch.softmax(outputs, dim=1).mean(0)
            _, pred = torch.max(avg_probs, 0)
            results.append([os.path.splitext(filenames[0])[0], pred.item()])
    
    df = pd.DataFrame(results, columns=['image_name', 'pred_label'])
    df.sort_values('image_name').to_csv('prediction.csv', index=False)
    print("已輸出prediction.csv 跟 train_history.csv")

if __name__ == '__main__':
    final_model = train_model()
    run_inference(final_model)