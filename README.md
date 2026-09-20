# SwastyaAI

A free, no-login, bilingual (English / Nepali) web platform for medical image screening across multiple diseases.

> **Research prototype for education only. Not a medical diagnosis.** Always consult a qualified doctor.

## What it does

Pick a disease module, upload an image, and get:

- Predicted class with probabilities
- Grad-CAM heatmap
- Uncertainty / low-confidence warning
- The model's measured performance, including external-dataset results

No accounts. Images are processed in memory and never stored.

## Disease modules (planned)

Diabetic retinopathy · Multi-disease fundus · Pneumonia / COVID-19 · Tuberculosis · Skin lesions · Brain tumor · Malaria · Breast cancer

Each is trained on 2+ public datasets and tested on a dataset it never saw during training.

## Tech stack

PyTorch (ResNet, EfficientNet, ConvNeXt, ViT) · Flask · ONNX · Render / Railway

## Quick start

```bash
git clone https://github.com/<your-username>/SwasthyaAI.git
cd SwasthyaAI
pip install -r requirements.txt
cd app && python app.py
```

Open http://127.0.0.1:5000

## Status

 In development. First milestone: Diabetic Retinopathy + Chest X-ray Pneumonia.

## Author

Bikram Chapagain


