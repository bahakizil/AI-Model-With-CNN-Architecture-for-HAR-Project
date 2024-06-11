import torch
from torchvision import models

model = models.vgg16(pretrained=True)  

model.eval()

print(model)
