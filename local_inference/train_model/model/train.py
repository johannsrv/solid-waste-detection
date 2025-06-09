from typing import Union, Tuple, List
import torch
from torch.nn import CrossEntropyLoss
from torchvision.models.mobilenetv3 import MobileNetV3
import torch.optim as optim
from torch.utils.data import DataLoader


class TrainModel:
    """
    A class for training a MobileNetV3 model with early stopping, learning rate scheduling, 
    and optional learning rate selection.

    Attributes:
        model (MobileNetV3): The neural network to train.
        dataloaders (List[DataLoader]): A list with training and validation DataLoaders.
        device (str): The device used for training ('cuda' or 'cpu').
        max_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait before early stopping if no improvement.
        criterion (CrossEntropyLoss): Loss function for classification.
        range_improvement (float): Minimum improvement required to consider a new best model.
    """

    def __init__(
        self,
        model: MobileNetV3,
        dataloaders: List[DataLoader],  
        device: str, 
        max_epochs: int = 10,
        patience: int = 5,
        range_improvement: float = 0.0
    ) -> None:
        """
        Initializes the training class with model, dataloaders, device, and hyperparameters.

        Args:
            model (MobileNetV3): The model to be trained.
            dataloaders (List[DataLoader]): Training and validation data loaders.
            device (str): 'cuda' or 'cpu'.
            max_epochs (int, optional): Max training epochs. Default is 10.
            patience (int, optional): Epochs without improvement before stopping. Default is 5.
            range_improvement (float, optional): Relative improvement threshold. Default is 0.0.
        """
        self.model = model.to(device)
        self.train_loader, self.val_loader, *_ = dataloaders
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.range_improvement = range_improvement
        self.criterion = CrossEntropyLoss()

    def _check_stopping(
        self,
        val_loss: float,
        epochs_no_improve: int,
        best_val_loss: float,
        best_model_state: dict
    ) -> Tuple[float, dict, int, bool]:
        """
        Checks for early stopping condition based on validation loss.

        Args:
            val_loss (float): Current validation loss.
            epochs_no_improve (int): Epochs without improvement.
            best_val_loss (float): Best validation loss so far.
            best_model_state (dict): Best model parameters.

        Returns:
            Tuple with updated best_val_loss, model state, counter, and stop flag.
        """
        stop = False
        if val_loss < best_val_loss * (1 - self.range_improvement):
            best_val_loss = val_loss
            best_model_state = self.model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= self.patience:
                print("Early stopping triggered.")
                stop = True

        return best_val_loss, best_model_state, epochs_no_improve, stop

    def _validation(self) -> Tuple[float, float]:
        """
        Runs evaluation on the validation set.

        Returns:
            Average loss and accuracy on the validation set.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs, loss = self._calculate_loss(images, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def _calculate_loss(self, images, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates loss for a given batch.

        Args:
            images (Tensor): Input images.
            labels (Tensor): Target labels.

        Returns:
            Tuple of model output and computed loss.
        """
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return outputs, loss

    def _move_to_device(self, images, labels) -> float:
        """
        Runs a single training step on one batch.

        Args:
            images (Tensor): Input images.
            labels (Tensor): Target labels.

        Returns:
            Loss value for the batch.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        _, loss = self._calculate_loss(images, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_model(self, lr: float, one_train: bool = False) -> Union[torch.nn.Module, Tuple]:
        """
        Trains the model using the specified learning rate.

        Args:
            lr (float): Learning rate.
            one_train (bool, optional): Return only model or also losses.

        Returns:
            Trained model or tuple (model, val_loss, train_loss).
        """
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            for images, labels in self.train_loader:
                total_train_loss += self._move_to_device(images, labels)

            avg_train_loss = total_train_loss / len(self.train_loader)
            val_loss, val_acc = self._validation()

            print(f"Epoch [{epoch}/{self.max_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            self.scheduler.step(val_loss)

            best_val_loss, best_model_state, epochs_no_improve, stop = self._check_stopping(
                val_loss, epochs_no_improve, best_val_loss, best_model_state
            )

            if stop:
                break

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.model if one_train else (self.model, val_loss, avg_train_loss)

    def selection_lr_model(self, list_lr: List[float]) -> torch.nn.Module:
        """
        Trains the model with multiple learning rates and selects the best one.

        Args:
            list_lr (List[float]): List of learning rates to test.

        Returns:
            torch.nn.Module: The best model based on validation loss.
        """
        best_model = None
        best_val_loss = float('inf')

        for lr in list_lr:
            print(f"\nTraining with learning rate: {lr}")
            model_train, val_loss, _ = self.train_model(lr=lr, one_train=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model_train

        return best_model
