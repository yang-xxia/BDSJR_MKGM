# 🏗️ BDSJR_MKGM

**Multimodal Knowledge Guided Model (MKGM)** for Joint Recognition of Bridge Defects and Structural Information.

## 🔍 Overview

Structural defect recognition is a crucial task in bridge inspection. However, existing methods typically focus solely on detecting surface defects while neglecting the structural context such as regions and components present in the background. This limitation hinders fine-grained defect understanding and leads to inaccurate defect attribution.
To address this challenge, **MKGM** introduces a multimodal knowledge-guided framework that jointly recognizes bridge defects along with their associated structural regions and components. It leverages image-text knowledge, concept graphs, and label co-occurrence patterns to guide feature propagation and semantic reasoning across hierarchical labels. The model incorporates the following components:

- 📷 **MambaVision** as the backbone network to extract both global visual features from inspection images  
- 🌐 **Multimodal knowledge graphs**, integrating visual object detection, label co-occurrence statistics, and text-based knowledge graph embeddings  
- 🔁 **Dual-channel GCNs** to propagate features and model hierarchical dependencies across structural and defect labels  
- 🧠 **Attention-based fusion** to adaptively integrate multimodal semantic information for robust joint prediction  

---

## 🖼️ Graphical Abstract

<p align="center">
  <img src="assets/graphical_abstract.png" width="700">
</p>

---

## 🔧 Environment Configuration

This project was developed and tested under the following environment:

- python = 3.8.2  
- torch = 2.2.0  
- torchvision = 0.17.0  
- numpy = 1.24.0  

💡 Optional: You can install dependencies with pip install -r requirements.txt or manually as shown below.

## 📁 Dataset Setup

The dataset contains **1,463 bridge inspection images**, each annotated with **hierarchical labels** across three levels:  
**structural region → component → defect type**

### 🔽 Download

You can download the full dataset from:

🔗 [Cloud Drive Download](https://ieeexplore.ieee.org/document/10546293) *(access code required)*

> 📌 **Note**: The extraction code is not publicly available. Please contact the authors for academic or collaborative use.  
> 📧 Contact Email: `your_email@example.com`

---

### 🗂 Directory Structure

After unzipping the dataset to the project root directory, it should have the following structure:

```bash
dataset/
├── Annotations/                 # raw label files for each image (multi-level: structural region/component/defect)
├── files/                       # preprocessed label CSVs
│   ├── classification_trainval.csv   # training/validation labels in multi-label format
│   └── classification_test.csv       # test labels in multi-label format
├── JPEGImages/                  # raw bridge inspection images
├── pool_pkls/                   # region-wise image features (Faster R-CNN)
├── co_occurrence_matrix.pkl     # label co-occurrence matrix (label correlation prior)
├── ent_emb.pkl                  # label embeddings from knowledge graph (used as semantic prior)
└── T-G-adj.pkl                  # adjacency matrix of textual graph (structural region–component–defect hierarchy)
```
---

### ✅ Usage Notes

- All annotations and features are **preprocessed** and ready to use.
- No additional conversion or annotation processing is required.


## 📥 Backbone Model: MambaVision-B-1K

The proposed **MKGM** model adopts **[MambaVision-B-1K](https://huggingface.co/nvidia/MambaVision-B-1K)** as the image feature extraction backbone.  
This pretrained model can be easily downloaded and used via [Hugging Face Transformers](https://huggingface.co) or the `timm` library.

```python
from timm import create_model
model = create_model('mambavision_b_1k', pretrained=True)







