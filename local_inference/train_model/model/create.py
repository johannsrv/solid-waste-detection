from torch.nn import Sequential, Linear, Hardswish, Dropout
from torchvision import models

class Model:
    """
    A class to build and customize a MobileNetV3-Large model for classification tasks.

    Attributes:
        device (str): The device on which the model will be loaded (e.g., 'cpu' or 'cuda').
        num_classes (int): The number of output classes for the classification task.
    """
    def __init__(self, device: str, num_classes: int =6):
        """
        Initializes the Model instance with the specified device and number of output classes.

        Args:
            device (str): The device to use for model computation.
            num_classes (int, optional): Number of classes for the final output layer. Defaults to 6.
        """
        self.device = device
        self.num_classes = num_classes

    def __generar_model(self):
        """
        Loads a pretrained MobileNetV3-Large model and prepares it for customization.
        Extracts the number of input features for the final classification layers.
        """
        # load a pre_trined mode 
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model = self.model.to(self.device)


        self.in_features = self.model.classifier[0].in_features


    def get_model(self):
        """
        Customizes the final classification layers of the pretrained model.

        Returns:
            torch.nn.Module: The modified MobileNetV3-Large model with custom classifier layers.
        """
        self.__generar_model()
        self.model.classifier = Sequential(
            Linear(self.in_features, 512),
            Hardswish(),
            Dropout(0.3),

            Linear(512, 256),
            Hardswish(),
            
            Linear(256, 128),
            Hardswish(),
            Dropout(0.4),
            
            Linear(128, self.num_classes)
        )

        self.model = self.model.to(self.device)

        return self.model

