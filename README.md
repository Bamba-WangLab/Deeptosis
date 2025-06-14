# Deeptosis
A Deep Learning-Based Tool for Automated Classification of Cell Death Modes from Brightfield Microscopy Images

## 1. Background

Cell death plays a pivotal role in various physiological and pathological processes. However, distinguishing between different cell death modes such as apoptosis and pyroptosis based on brightfield microscopy images remains challenging. Deeptosis is developed to automate this task using deep learning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/12402cf0-b395-42c0-89f5-1bb21fbc1e9f" alt="Apoptosis" width="350"/>
  <img src="https://github.com/user-attachments/assets/8485c42a-9b7f-4090-8f76-aa97bf6708f9" alt="Pyroptosis" width="350"/>
</p>

<p align="center"><em>Left: Apoptosis morphology &nbsp;&nbsp;&nbsp;&nbsp; Right: Pyroptosis morphology</em></p>

## 2. Model Architecture

Deeptosis employs a Vision Transformer (ViT) architecture to classify cell death modes directly from brightfield microscopy images. Unlike conventional CNNs, ViT models capture global context via self-attention, which is particularly useful for subtle morphological differences in apoptosis and pyroptosis.

<p align="center">
  <img src="https://github.com/user-attachments/assets/41274370-e8e4-4fa5-b800-4b2638fe304a" width="700"/>
</p>

<p align="center"><em>Overview of the Vision Transformer (ViT) architecture used in Deeptosis.</em></p>

## 3. Performance on the Test Set

To evaluate the model's performance, the dataset was randomly split into training, validation, and test sets in an 8:1:1 ratio. The test set, although not an external dataset, was held out from training and used exclusively for evaluation.

Deeptosis achieved an overall accuracy of **91.7%** on the test set, with high class-wise precision and recall. The model demonstrates strong discriminative ability across apoptosis and pyroptosis.

**Key performance metrics:**
- Accuracy: **91.7%**
- Macro-average Precision: **91%**
- Macro-average Recall: **92%**
- Macro-average F1-score: **91%**
- Multi-class Macro-AUC: **0.985**

**Per-class metrics:**
| Class       | Precision | Recall | F1-score | Support |
|-------------|-----------|--------|----------|---------|
| Apoptosis   | 0.90      | 0.95   | 0.92     | 672     |
| Pyroptosis  | 0.94      | 0.92   | 0.93     | 1203    |
| Other       | 0.90      | 0.88   | 0.89     | 782     |

<p align="center">
  <img src="https://github.com/user-attachments/assets/1717d36e-ad28-4b39-ab99-90cbcb054bbe" alt="Confusion Matrix" width="400"/>
  <img src="https://github.com/user-attachments/assets/9311c631-20ec-4a2e-85b9-86a4ab1fce97" alt="ROC Curve" width="400"/>
</p>

<p align="center"><em>Left: Confusion matrix showing class-wise accuracy. Right: Multi-class ROC curves with AUCs > 0.98 across all categories.</em></p>

## 4. Usage

This section describes how to use the final trained Deeptosis model to perform cell segmentation and classification on microscopy images.

---

### 🔧 Step 1: Set up environment and install dependencies

We recommend using Python 3.9 and installing the required packages via `pip`:

```bash
pip install torch==2.6.0 torchvision==0.21.0
pip install timm==1.0.15
pip install cellpose==3.1.1.2
pip install numpy==2.0.2 pillow==11.2.1 scipy==1.13.1 tqdm==4.67.1
pip install opencv-python-headless==4.11.0.86
```

> ✅ Tested environment:
>
> * Python 3.9.12
> * PyTorch 2.6.0 (CUDA 12.4)
> * Cellpose 3.1.1.2
> * timm 1.0.15

---

### 📦 Step 2: Download final trained model

Download the final trained ViT model weights from the GitHub release:

🔗 **[best\_weight\_VitTrans.pth](https://github.com/Bamba-WangLab/Deeptosis/releases/download/v1.0-model/best_weight_VitTrans.pth)**

Place this file in the **same folder** as the script `predict_cells.py`.

---

### 📂 Step 3: Download example test dataset and script

You can find the test folder and script in the repository:

* [`test/`](https://github.com/Bamba-WangLab/Deeptosis/tree/main/test): contains sample input images
* [`predict_cells.py`](https://github.com/Bamba-WangLab/Deeptosis/blob/main/test/predict_cells.py): main prediction pipeline script

Make sure your folder structure looks like this:

```
project_folder/
├── best_weight_VitTrans.pth
├── predict_cells.py
└── test/
    ├── image1.png
    ├── image2.jpg
    └── ...
```

---

### ▶️ Step 4: Run the prediction script

Use the following command to run the prediction:

```bash
python predict_cells.py \
  --input_dir test \
  --use_gpu \
  --threshold 0.5
```

---

### ✅ Output files

After execution, the following results will be generated:

* 📸 **Boxed images** saved in `test/boxed/` with bounding boxes:

  * 🟩 Apoptosis (green)
  * 🔴 Pyroptosis (red)

* 📄 **CSV file**: `results.csv` with format:

```
filename, cell_id, x1, y1, x2, y2, pred_class, class_name, confidence
```

Example row:

```
image1.png, 12, 34, 58, 102, 130, 1, pyroptosis, 0.9821
```

---

