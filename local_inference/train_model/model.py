import torch
from torch.nn import Sequential, Linear, ReLu, Dropout
from torchvision import models

class Model:
    def __init__(self, num_classes: int =6):
        self.device = torch.device(
            "cuda" if 
            torch.cuda.is_available() 
            else "cpu")
        
        self.num_classes = num_classes

    def get_model(self):
        # load a pre_trined mode 
        model = models.mobilnet_v3_large(pretrained= True)
        model = model.to(self.device)

        # adds 4 new Linear layers to the model 
        in_features = model.classifier[3].in_features

        model.classifier = Sequential(
            Linear(in_features, 512),
            ReLu(),
            Dropout(0.3),

            Linear(512, 256),
            ReLu(),
            
            Linear(256, 128),
            ReLu(),
            Dropout(0.4),
            
            Linear(128, self.num_classes)
        )

        return model


