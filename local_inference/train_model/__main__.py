from typing import List, Optional
import os 
import torch

from data.daltaloader import CreateDataLoader
from model.create import Model
from model.predict import Prediction
from model.train import TrainModel

def optimizar_model(model):
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
    # os.makedirs("results_test", exist_ok=True)
    list_lr = [1e-3, 1e-4, 1e-5, 1e-7]
    main(list_lr)