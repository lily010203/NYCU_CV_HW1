# Ensemble
import torch, os, gc, glob, pandas as pd
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast

# 路徑
RESNET_PATH = '/kaggle/input/datasets/liupeilinlily/my-checkpoints/best_resnet.pth'
RESNEXT_PATH = '/kaggle/input/datasets/liupeilinlily/my-checkpoints/best_resnext.pth'
TEST_DIR = '/kaggle/input/datasets/liupeilinlily/computer-vision-hw1/data/test'
NUM_CLASSES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型
def get_model(arch_fn, path):
    model = arch_fn(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, NUM_CLASSES))
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

print("加載模型")
model_a = get_model(models.resnet101, RESNET_PATH)         # ResNet-101
model_b = get_model(models.resnext101_32x8d, RESNEXT_PATH) # ResNeXt-101

# 十點裁切 TTA
tta_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(c) for c in crops])),
])

class FlatTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        return self.transform(Image.open(img_path).convert('RGB')), os.path.basename(img_path)

# Ensemble Loop
test_loader = DataLoader(FlatTestDataset(TEST_DIR, tta_transform), batch_size=1)
results = []

weight_resnet = 0.5  
weight_resnext = 0.5 

print("Ensemble + Ten-Crop TTA")
with torch.no_grad():
    for inputs, filenames in tqdm(test_loader):
        # inputs shape: [1, 10, 3, 224, 224]
        bs, n_crops, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w).to(device)
        
        with autocast(device_type='cuda'):
            # 模型A的10張裁切平均機率
            out_a = torch.softmax(model_a(inputs), dim=1).mean(0)
            # 模型B的10張裁切平均機率
            out_b = torch.softmax(model_b(inputs), dim=1).mean(0)
            
            final_probs = (out_a * weight_resnet) + (out_b * weight_resnext)
            
        pred = torch.argmax(final_probs).item()
        results.append([os.path.splitext(filenames[0])[0], pred])

# 存檔
df = pd.DataFrame(results, columns=['image_name', 'pred_label']).sort_values('image_name')
df.to_csv('ensemble_prediction.csv', index=False)
print("已輸出ensemble_prediction.csv")
