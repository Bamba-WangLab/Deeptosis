import os
import csv
from glob import glob
from sklearn.model_selection import train_test_split

# 明确各类别的路径和标签编号
class1_dir = "D:/Deeptosis/0"
class2_dir = "D:/Deeptosis/zyy/1-zyy"
class3_dir = "D:/Deeptosis/Huang/Apoptosis_all/extracted_cells/2"
classes = [
    ("class1", class1_dir, 0),
    ("class2", class2_dir, 1),
    ("class3", class3_dir, 2),
]

output_dir = "D:\Deeptosis"
os.makedirs(output_dir, exist_ok=True)

img_label_list = []
for cls_name, cls_path, label in classes:
    img_paths = glob(os.path.join(cls_path, "*.png"))  
    for img in img_paths:
        img_label_list.append([img, 0, 0, 0, 0, label, label, label])

# 分层采样划分
img_labels = [row[7] for row in img_label_list]
trainval, test = train_test_split(img_label_list, test_size=0.1, random_state=42, stratify=img_labels)
trainval_labels = [row[7] for row in trainval]
train, val = train_test_split(trainval, test_size=0.1111, random_state=42, stratify=trainval_labels)

# 保存为csv
for name, split in zip(['train', 'val', 'test'], [train, val, test]):
    with open(os.path.join(output_dir, f'{name}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(split)

print("所有CSV已生成，三类图片路径分别独立采集完成。")