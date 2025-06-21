---
license: other
license_name: nvclv1
license_link: LICENSE
datasets:
- ILSVRC/imagenet-1k
pipeline_tag: image-feature-extraction
---


[**MambaVision: A Hybrid Mamba-Transformer Vision Backbone**](https://arxiv.org/abs/2407.08083).

## Model Overview

We have developed the first hybrid model for computer vision which leverages the strengths of Mamba and Transformers. Specifically, our core contribution includes redesigning the Mamba formulation to enhance its capability for efficient modeling of visual features. In addition, we conducted a comprehensive ablation study on the feasibility of integrating Vision Transformers (ViT) with Mamba. Our results demonstrate that equipping the Mamba architecture with several self-attention blocks at the final layers greatly improves the modeling capacity to capture long-range spatial dependencies. Based on our findings, we introduce a family of MambaVision models with a hierarchical architecture to meet various design criteria.

## Model Performance

MambaVision demonstrates a strong performance by achieving a new SOTA Pareto-front in
terms of Top-1 accuracy and throughput. 

<p align="center">
<img src="https://github.com/NVlabs/MambaVision/assets/26806394/79dcf841-3966-4b77-883d-76cd5e1d4320" width=70% height=70% 
class="center">
</p>


## Model Usage

It is highly recommended to install the requirements for MambaVision by running the following:


```Bash
pip install mambavision
```

For each model, we offer two variants for image classification and feature extraction that can be imported with 1 line of code. 

### Image Classification

In the following example, we demonstrate how MambaVision can be used for image classification. 

Given the following image from [COCO dataset](https://cocodataset.org/#home)  val set as an input:


<p align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/64414b62603214724ebd2636/4duSnqLf4lrNiAHczSmAN.jpeg" width=70% height=70% 
class="center">
</p>


The following snippet can be used for image classification:

```Python
from transformers import AutoModelForImageClassification
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests

model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)

# eval mode for inference
model.cuda().eval()

# prepare image for the model
url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
image = Image.open(requests.get(url, stream=True).raw)
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions

transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)

inputs = transform(image).unsqueeze(0).cuda()
# model inference
outputs = model(inputs)
logits = outputs['logits'] 
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

The predicted label is ```brown bear, bruin, Ursus arctos.```

### Feature Extraction

MambaVision can also be used as a generic feature extractor. 

Specifically, we can extract the outputs of each stage of model (4 stages) as well as the final averaged-pool features that are flattened. 

The following snippet can be used for feature extraction:

```Python
from transformers import AutoModel
from PIL import Image
from timm.data.transforms_factory import create_transform
import requests

model = AutoModel.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)

# eval mode for inference
model.cuda().eval()

# prepare image for the model
url = 'http://images.cocodataset.org/val2017/000000020247.jpg'
image = Image.open(requests.get(url, stream=True).raw)
input_resolution = (3, 224, 224)  # MambaVision supports any input resolutions

transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)
inputs = transform(image).unsqueeze(0).cuda()
# model inference
out_avg_pool, features = model(inputs)
print("Size of the averaged pool features:", out_avg_pool.size())  # torch.Size([1, 640])
print("Number of stages in extracted features:", len(features)) # 4 stages
print("Size of extracted features in stage 1:", features[0].size()) # torch.Size([1, 80, 56, 56])
print("Size of extracted features in stage 4:", features[3].size()) # torch.Size([1, 640, 7, 7])
```


### License: 

[NVIDIA Source Code License-NC](https://huggingface.co/nvidia/MambaVision-T-1K/blob/main/LICENSE)