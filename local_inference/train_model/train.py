import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load a pre_trined and configure mode 
model = models.mobilnet_v3_large(pretrained= True)

num_classes = 6
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)