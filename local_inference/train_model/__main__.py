from typing import List, Optional
import os 
import torch

from data.daltaloader import CreateDataLoader
from model.create import Model
from model.predict import Prediction
from model.train import TrainModel

def optimizar_model(model):
    """
    Applies dynamic quantization to reduce the model size and improve inference performance.

    The function:
    - Moves the model to CPU.
    - Applies dynamic quantization to `torch.nn.Linear` layers using 8-bit integers (qint8).
    - Converts the quantized model into a TorchScript scripted model for optimized deployment.

    Args:
        model (torch.nn.Module): The trained PyTorch model to optimize.

    Returns:
        torch.jit.ScriptModule: The optimized scripted model.
    """
    ...
    # convert the model to evaluation mode
    model = model.to('cpu')

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # convert the quantized model to a scripted model
    scripted_model = torch.jit.script(quantized_model)

    return scripted_model

def save_model(model, path: str, save_model_weights: bool ) -> None:
    """
    Saves the model in different formats depending on the configuration.

    Args:
        model (torch.nn.Module or torch.jit.ScriptModule): The trained (and optionally scripted) model.
        path (str): The base path to save the model files.
        save_model_weights (bool): If True, only the model weights and scripted model are saved.
                                   If False, the entire model and scripted version are saved.

    Saves:
        - Model weights in `<path>_weights_quantized.pth`
        - Scripted model (if applicable) in `<path>_scripted_quantized.pth`
        - Entire model in `<path>_quantized.pth`
        - Scripted model (if applicable) in `<path>_scripted.pth`
    """
    complement_path = [
        "weights_quantized", 
        "scripted_quantized", 
        "quantized", 
        "scripted"]

    # save the model or save the model weights
    if save_model_weights:
        # Save the model weights
        torch.save(
            model.state_dict(),
            f"{path}_{complement_path[0]}.pth"
        )
        
        # Save the scripted model
        if isinstance(model, torch.jit.ScriptModule):
            model.save(f"{path}_{complement_path[1]}.pth")
    
    else:
        # Save the entire model
        torch.save(model, f"{path}_{complement_path[2]}.pth")

        if isinstance(model, torch.jit.ScriptModule):
            model.save(f"{path}_{complement_path[3]}.pth")

def main(
        list_lr: Optional[List[float]] = None,
        lr: float = 1e-3, 
        save_model_weights: bool = True) -> None:
    """
    Main function for training, optimizing, evaluating, and saving a classification model.

    This function:
    - Initializes the model and data loaders.
    - Trains the model using a fixed or multiple learning rates.
    - Optimizes the model using dynamic quantization and TorchScript scripting.
    - Evaluates the model on the test set and generates visual performance reports.
    - Saves the model in different formats.

    Args:
        list_lr (Optional[List[float]], optional): List of learning rates for selection. 
            If None, the model is trained using the provided `lr` value. Default is None.
        lr (float, optional): Learning rate for training if `list_lr` is not used. Default is 1e-3.
        save_model_weights (bool, optional): If True, saves only the model weights and scripted model.
            If False, saves the full model. Default is True.
    """
    
    path = "detecction_recycling.v1i.folder"

    test = Prediction()

    # Set the device for PyTorch
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() 
        else "cpu")
    
    # Define the paths
    path = "detecction_recycling.v1i.folder"

    # Initialize the modelss
    model = Model(
        device=device, 
        num_classes=6
    ).get_model()

    # Initialize the Daltaloadera
    daltaloader = CreateDataLoader(path)
    list_data_set = daltaloader.datoloader_generar()
    
    train_model = TrainModel(
        model=model,
        dataloaders=list_data_set,
        device=device,
        max_epochs= 100
    )

    if list_lr is None:
        # Train the model with a single learning rate
        model = train_model.train(lr=lr)
    
    else:
        model = train_model.selection_lr_model(list_lr)
    
    # Obtimize the model
    model = optimizar_model(model)
    
    # Generar test and report the model
    test.test_model(
        model=model,
        test_loader=list_data_set[2],
        clases=daltaloader.classes,
        number_samper=5
    )

    test.generate_visual_metrics_report(
        clases=daltaloader.classes
    )

    # Save the model
    save_model(model, path, save_model_weights)

if __name__ == "__main__":
    list_lr = [1e-3, 1e-4, 1e-5, 1e-7]
    main(list_lr)