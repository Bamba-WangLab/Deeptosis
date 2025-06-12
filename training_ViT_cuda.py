import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
from PIL import Image
import timm
from tqdm import tqdm

# ---- Custom Path Configuration ----
DATA_FOLDER = 'data/Deeptosis/label'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')
TEST_CSV = os.path.join(DATA_FOLDER, 'test.csv')
WEIGHTS_DIR = 'Vit_Transformer_weights/GorC_3_label/'
os.makedirs(WEIGHTS_DIR, exist_ok=True)
RESULT_NPY = 'results/VitTrans_result.npy'
os.makedirs('results', exist_ok=True)

df = pd.read_csv(TRAIN_CSV, header=None)
bad = []
for idx, row in df.iterrows():
    if not os.path.exists(row[0]):
        bad.append(row[0])
print("Number of Non-existent Images：", len(bad))
if bad:
    print("Examples of Non-existent Images：", bad[:5])

# ---- Dataset Definition ----
class MultiDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        y = self.data_frame.iloc[idx, 7]
        if self.transform:
            image = self.transform(image)
        return image, y, img_name

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
NUM_CLASSES = 3
PRETRAINED_PATH = 'data/Deeptosis/pytorch_model_vit_base_patch16_224.pth'
assert os.path.exists(PRETRAINED_PATH), "Pretrained Weight File Does Not Exist, Please Download First!"

model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,  
    num_classes=NUM_CLASSES
)
checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
if 'model' in checkpoint:
    checkpoint = checkpoint['model']
for key in list(checkpoint.keys()):
    if key.startswith('head'):
        del checkpoint[key]
model.load_state_dict(checkpoint, strict=False)
model.to(device)

# ---- Loss Function, Optimizer, and Epochs ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)
num_epochs = 100

def train():
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            dataloader = tqdm(dataloaders[phase], desc=f'{phase} epoch {epoch+1}', leave=False)
            for batch_idx, (inputs, labels, _) in enumerate(dataloader):
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
                dataloader.set_postfix(loss=running_loss/((batch_idx+1)*inputs.size(0)))
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'validation' and epoch_acc > best_acc:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'best_weight_VitTrans.pth'))
                best_acc = epoch_acc
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, 'final_weight_VitTrans.pth'))

def test():
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, 'best_weight_VitTrans.pth'), map_location=device))
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
