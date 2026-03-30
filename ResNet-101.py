# ResNet-101
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

# 隱藏警告
warnings.filterwarnings("ignore")

# 參數、路徑
DATA_DIR = '/kaggle/input/datasets/liupeilinlily/computer-vision-hw1/data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

NUM_CLASSES = 100
BATCH_SIZE_PER_GPU = 32  
EPOCHS = 50 
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()
total_batch_size = BATCH_SIZE_PER_GPU * gpu_count if gpu_count > 0 else BATCH_SIZE_PER_GPU

# 資料處理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # 自動增強策略
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
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_id = os.path.basename(img_path)
        if self.transform: image = self.transform(image)
        return image, image_id

def get_loaders():
    my_class_to_idx = {str(i): i for i in range(NUM_CLASSES)}
    def fix_ds(ds):
        ds.class_to_idx = my_class_to_idx
        ds.samples = [(p, my_class_to_idx[os.path.basename(os.path.dirname(p))]) for p, _ in ds.samples]
        return ds

    train_ds = fix_ds(datasets.ImageFolder(TRAIN_DIR, data_transforms['train']))
    val_ds = fix_ds(datasets.ImageFolder(VAL_DIR, data_transforms['val']))
    
    return {
        'train': DataLoader(train_ds, batch_size=total_batch_size, shuffle=True, num_workers=0, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=total_batch_size, shuffle=False, num_workers=0),
    }

# 主程式
def main():
    loaders = get_loaders()
    
    print("載入 ResNet-101 (約 44M params)...")
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    model = model.to(device)
    if gpu_count > 1: model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    best_acc = 0.0
    history = [] # 初始化歷史紀錄列表

    for epoch in range(EPOCHS):
        epoch_log = {'epoch': epoch + 1} # 建立Epoch的紀錄字典
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            
            pbar = tqdm(loaders[phase], desc=f'Epoch {epoch+1}/{EPOCHS} {phase}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)
            epoch_loss = running_loss / len(loaders[phase].dataset)
            
            # 把數值存入字典
            epoch_log[f'{phase}_acc'] = epoch_acc.item()
            epoch_log[f'{phase}_loss'] = epoch_loss
            
            print(f'{phase} Acc: {epoch_acc:.4f} Loss: {epoch_loss:.4f}')
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                state = model.module.state_dict() if gpu_count > 1 else model.state_dict()
                torch.save(state, 'best_model_resnet101.pth')
                print(f"目前發現的最佳模型：{best_acc:.4f}")

        # 每個Epoch結束後儲存CSV ->才不用全部重跑
        history.append(epoch_log)
        pd.DataFrame(history).to_csv('train_history.csv', index=False)

        # 每輪結束後強制清理記憶體防止Kernel Died  ->但kaggle一定程度之後還是會死掉
        scheduler.step()
        gc.collect()
        torch.cuda.empty_cache()

    # Ten-Crop TTA
    print("\nTen-Crop TTA 推論")
    if os.path.exists('best_model_resnet101.pth'):
        state_dict = torch.load('best_model_resnet101.pth')
        if gpu_count > 1: model.module.load_state_dict(state_dict)
        else: model.load_state_dict(state_dict)
    
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
        for inputs, filenames in tqdm(test_loader, desc='Final Inference'):
            bs, n_crops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w).to(device)
            
            with autocast(device_type='cuda'):
                outputs = model(inputs)
            
            avg_probs = torch.softmax(outputs, dim=1).mean(0)
            _, pred = torch.max(avg_probs, 0)
            results.append([os.path.splitext(filenames[0])[0], pred.item()])
    
    df = pd.DataFrame(results, columns=['image_name', 'pred_label'])
    df.sort_values(by='image_name', inplace=True)
    df.to_csv('prediction.csv', index=False)
    print("已輸出prediction.csv 跟 train_history.csv")

if __name__ == '__main__':
    main()