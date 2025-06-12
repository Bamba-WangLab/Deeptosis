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

