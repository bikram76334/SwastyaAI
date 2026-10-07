"""
Image preprocessing for the diabetic retinopathy (DR) model — the exact
same pipeline used in bo-dr-2024.ipynb (cells 4-5), so an image gets
treated identically whether it runs in the training notebook or here in
the web app.

Pipeline: crop black border -> pad to square -> resize to 512x512 ->
Ben Graham illumination correction -> CLAHE contrast enhancement ->
scale to [0,1] -> normalise with ImageNet mean/std.

No augmentation here on purpose: random flips/rotation/jitter in the
notebook are train-only (cell 5, train_aug) and must never be applied at
inference time, or predictions stop matching the notebook's test numbers.

Same pattern for other modules: copy this file, keep the function
names, swap IMG_SIZE 
 the preprocessing steps for what that model's
notebook actually used.
"""

import cv2
import numpy as np
import torch

IMG_SIZE = 512  #must match IMG in the training notebook

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def crop_black(img):
    """Removes the black border around the retina (notebook cell 4)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 10)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make_square(img):
    """Pads the short side so the retina isn't stretched by resize."""
    h, w = img.shape[:2]
    d = abs(h - w)
    a, b = d // 2, d - d // 2
    if h > w:
        return cv2.copyMakeBorder(img, 0, 0, a, b, cv2.BORDER_CONSTANT, value=0)
    return cv2.copyMakeBorder(img, a, b, 0, 0, cv2.BORDER_CONSTANT, value=0)


def preprocess_array(img_rgb):
    """Crop -> square -> resize -> Ben Graham -> CLAHE on an already
    decoded RGB uint8 array. Returns a uint8 (IMG_SIZE, IMG_SIZE, 3)
    array — identical to preprocess() in the training notebook."""
    img = make_square(crop_black(img_rgb))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    #Ben Graham illumination correction: 4*I - 4*blur + 128
    sigma = IMG_SIZE / 2 / 30
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    #CLAHE on the L channel only (keeps colour, boosts local contrast)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


def preprocess_bytes(image_bytes):
    """Takes the raw bytes of an uploaded file (request.files['image']
    .read()) and returns a normalised (1, 3, IMG_SIZE, IMG_SIZE) tensor,
    ready to feed straight into the model."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not read this file as an image.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = preprocess_array(img_rgb)

    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    x = (x - MEAN) / STD
    return x.unsqueeze(0)  #add the batch dimension