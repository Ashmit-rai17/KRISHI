# model_utils.py
import torch
from torchvision import transforms
from PIL import Image
import json
import os

IMG_SIZE = 224
MODEL_DIR = "saved_models"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_labels():
    p = os.path.join(MODEL_DIR, "labels.json")
    with open(p, "r") as f:
        labels = json.load(f)
    # ensure int keys -> str classes
    return {int(k): v for k, v in labels.items()}

def load_model(device="cpu", quantized=True):
    """
    Loads the quantized TorchScript model if present, else scripted model, else state_dict model.
    """
    # prefer quantized scripted
    qpath = os.path.join(MODEL_DIR, "best_model_quantized.pt")
    spath = os.path.join(MODEL_DIR, "best_model_scripted.pt")
    if quantized and os.path.exists(qpath):
        model = torch.jit.load(qpath, map_location=device)
        model.eval()
        return model
    if os.path.exists(spath):
        model = torch.jit.load(spath, map_location=device)
        model.eval()
        return model
    # fallback: load state_dict into a mobilenet and return
    from torchvision import models
    import torch.nn as nn
    model = models.mobilenet_v2(pretrained=False)
    sd = os.path.join(MODEL_DIR, "best_model.pth")
    labels = load_labels()
    num_classes = len(labels)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(sd, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, img_pil, topk=1):
    x = transform(img_pil).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        out = model(x)
        # if scripted quantized model returns Tensor
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.nn.functional.softmax(out, dim=1)
        top_probs, top_idx = probs.topk(topk, dim=1)
        return top_probs.squeeze(0).tolist(), top_idx.squeeze(0).tolist()
