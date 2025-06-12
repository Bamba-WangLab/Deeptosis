import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
from PIL import Image
from efficientnet_pytorch import EfficientNet

# ---- Custom Path Configuration ----
DATA_FOLDER = 'D:/Deeptosis'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')
TEST_CSV = os.path.join(DATA_FOLDER, 'test.csv')
WEIGHTS_DIR = 'Efficientnet_B3_weights/GorC_3_label/'
os.makedirs(WEIGHTS_DIR, exist_ok=True)
RESULT_NPY = 'results/E_B3_result.npy'
os.makedirs('results', exist_ok=True)

df = pd.read_csv('D:/Deeptosis/train.csv', header=None, encoding='gbk')
bad = []
for idx, row in df.iterrows():
    if not os.path.exists(row[0]):
        bad.append(row[0])
print("Number of Non-existent Images: ", len(bad))
if bad:
    print("Examples of Non-existent Images: ", bad[:5])

# ---- Dataset Definition ----
class MultiDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None, encoding='gbk')
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        if idx % 100 == 0:
            print(f'Loading {idx}-th Image: {img_name}')
        image = Image.open(img_name).convert('RGB')
        y1 = self.data_frame.iloc[idx, 7]
        if self.transform:
            image = self.transform(image)
        return image, y1, img_name

# ---- Transformation Settings ----
normalize = transforms.Normalize(mean=[0.5115, 0.5115, 0.5115], std=[0.1316, 0.1316, 0.1316])
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(), normalize
])
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), normalize
])

# ---- DataLoader Configuration ----
image_datasets = {
    'train': MultiDataset(TRAIN_CSV, transform=train_transforms),
    'validation': MultiDataset(VAL_CSV, transform=val_test_transforms),
    'test': MultiDataset(TEST_CSV, transform=val_test_transforms)
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=0, pin_memory=False),
    'validation': DataLoader(image_datasets['validation'], batch_size=8, shuffle=False, num_workers=0, pin_memory=False),
    'test': DataLoader(image_datasets['test'], batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
}

# ---- Model and Device Configuration ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNet.from_name('efficientnet-b3', num_classes=3)
weight_path = 'D:/Deeptosis/efficientnet-b3-5fb5a3c3.pth'
assert os.path.exists(weight_path), "Pretrained Weight File Does Not Exist, Please Download First!"
pretrained_dict = torch.load(weight_path, map_location='cpu')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('_fc.')}
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.to(device)

# ---- Loss Function, Optimizer, and Epochs ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
num_epochs = 100

def adjust_learning_rate(optimizer, epoch):
    lr = 0.1 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        adjust_learning_rate(optimizer, epoch)
        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            for batch_idx, (inputs, labels, _) in enumerate(dataloaders[phase]):
                if batch_idx % 10 == 0:
                    print(f'    ‌{phase} Dataset - Batch {batch_idx} Loaded')
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'validation' and epoch_acc > best_acc:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'best_weight_E_B3.pth'))
                best_acc = epoch_acc
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'final_weight_E_B3.pth'))

def test():
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'best_weight_E_B3.pth'), map_location=device))
    model.eval()
    softM = nn.Softmax(1)
    labels_result, preds_result, outputs_result, img_names = [], [], [], []
    with torch.no_grad():
        for inputs, labels, img_name in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            labels_result.append(labels.cpu().numpy())
            preds_result.append(preds.cpu().numpy())
            outputs_result.append(softM(outputs).cpu().numpy())
            img_names.extend(img_name)
    labels_result = np.hstack(labels_result)
    preds_result = np.hstack(preds_result)
    outputs_result = np.vstack(outputs_result)
    img_names = np.array(img_names).reshape(-1,1)
    labels_result = labels_result.reshape(-1,1)
    preds_result = preds_result.reshape(-1,1)
    final_result = np.hstack([img_names, labels_result, preds_result, outputs_result])
    np.save(RESULT_NPY, final_result)
    print(f"Test Set Prediction Results Saved to {RESULT_NPY}")

if __name__ == '__main__':
    train()
    test()
