from typing import Union, Tuple
import torch
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import MobileNetV3
import torch.optim as optim
from torch.utils.data import DataLoader

class TrainModel:
    """
    Class for training a deep learning model with early stopping and learning rate selection.

    Attributes:
        model (MobileNetV3): The neural network model to be trained.
        dataloaders (list[DataLoader]): A list containing training and validation DataLoaders.
        device (str): The device to run the training on (e.g., 'cpu' or 'cuda').
        max_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait without improvement before early stopping.
        range_improvement (float): Minimum relative improvement required to reset early stopping counter.
        criterion (CrossEntropyLoss): Loss function used during training.
    """
    def __init__(
        self,
        model: MobileNetV3,
        dataloaders: list[DataLoader],  
        device: str, 
        max_epochs: int = 10,
        patience: int = 5,
        range_improvement: float = 0.0) -> None:
        """
        Initializes the training setup with model, dataloaders, training parameters and loss function.

        Args:
            model (MobileNetV3): The model to be trained.
            dataloaders (list[DataLoader]): A list containing the training and validation DataLoaders.
            device (str): The computation device ('cuda' or 'cpu').
            max_epochs (int, optional): The maximum number of training epochs. Default is 10.
            patience (int, optional): Number of epochs with no improvement before early stopping. Default is 5.
            range_improvement (float, optional): Threshold for relative improvement to consider as significant. Default is 0.05.
        """
        
        self.range_improvement = range_improvement
        self.model = model.to(device)
        self.train_loader, self.val_loader, *_ = dataloaders
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.criterion = CrossEntropyLoss()

    def _check_stopping(
            self, 
            val_loss: float,
            epochs_no_improve: int, 
            best_val_loss: float, 
            best_model_state: dict) -> Tuple:
        """
        Checks whether early stopping criteria are met.

        Args:
            val_loss (float): Current validation loss.
            epochs_no_improve (int): Number of consecutive epochs without significant improvement.
            best_val_loss (float): Best validation loss seen so far.
            best_model_state (dict): Current best model state.

        Returns:
            tuple: Updated (best_val_loss, best_model_state, epochs_no_improve, stop flag).
        """
        adjusted_val_loss = val_loss * (1 - self.range_improvement)
        stop = False

        if adjusted_val_loss < best_val_loss:
            best_val_loss = adjusted_val_loss
            best_model_state = self.model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= self.patience:
                print("Early stopping triggered.")
                stop = True

        return best_val_loss, best_model_state, epochs_no_improve, stop

    def _validation(self) -> Tuple:
        """
        Runs validation on the model.

        Returns:
            tuple: Average validation loss and validation accuracy.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = (
                    images.to(self.device),
                    labels.to(self.device))
                
                outputs, loss = self._calculate_loss(images, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / len(self.val_loader)
        val_accuracy = 100 * correct / total
        return avg_loss, val_accuracy
    
    def _calculate_loss(self, images, labels) -> Tuple:
        """
        Computes the loss for a batch of data.

        Args:
            images (Tensor): Input images.
            labels (Tensor): Ground truth labels.

        Returns:
            tuple: Model outputs and computed loss.
        """
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def _move_to_device(self, images, labels) -> float:
        """
        Moves input data to the appropriate device and performs a single training step.

        Args:
            images (Tensor): Input images.
            labels (Tensor): Ground truth labels.

        Returns:
            float: Loss value for the batch.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        _, loss = self._calculate_loss(images, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_model(
            self, 
            lr: float, 
            one_train: bool = False) -> Union[torch.nn.Module, Tuple]:
        """
        Trains the model with the specified learning rate, applying early stopping.

        Args:
            lr (float): Learning rate to use for training.
            one_train (bool, optional): If True, only returns the trained model.
                                         If False, returns model, validation loss, and training loss.

        Returns:
            Union[torch.nn.Module, tuple]: Trained model (and optionally val/train losses).
        """
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss = 0.0

            # Corregido: Eliminar bucle anidado
            for images, labels in self.train_loader:
                batch_loss = self._move_to_device(images, labels)
                train_loss += batch_loss  # Acumular pérdida

            avg_train_loss = train_loss / len(self.train_loader)
            val_loss, val_acc = self._validation()

            print(f"Epoch [{epoch}/{self.max_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            # Corregido: Manejo de early stopping
            (best_val_loss, 
             best_model_state, 
             epochs_no_improve, 
             stop) = self._check_stopping(
                val_loss, 
                epochs_no_improve, 
                best_val_loss, 
                best_model_state)
            
            if stop:
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        if one_train:
            return self.model
        return self.model, val_loss, avg_train_loss
    
    def selection_lr_model(self, list_lr: list[float]) -> torch.nn.Module:
        """
        Trains the model using a list of learning rates and returns the best-performing model.

        Args:
            list_lr (list[float]): List of learning rates to evaluate.

        Returns:
            torch.nn.Module: The model with the lowest validation loss.
        """
        best_model = None
        best_val_loss = float('inf')
        
        for lr in list_lr:
            print("training with learning rate:", lr)
            model_train, val_loss, _ = self.train_model(lr=lr, one_train=False)
            if val_loss < best_val_loss:
                best_model = model_train
                best_val_loss = val_loss

        return best_model