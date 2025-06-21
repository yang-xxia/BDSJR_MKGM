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

## 🧠 Graphical Abstract

<p align="center">
  <img src="assets/graphical_abstract.png" width="700">
</p>

