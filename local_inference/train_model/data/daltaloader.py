import os

import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm



class CreateDataLoader:
    """
    A class to generate PyTorch DataLoaders for training, validation, and testing image datasets.
    Also computes the mean and standard deviation for normalization from the training dataset.

    Attributes:
        data_dir (str): Path to the root dataset directory containing 'train', 'val', and 'test' folders.
        batch_size (int): Number of samples per batch. Default is 32.
        num_workers (int): Number of subprocesses to use for data loading. Default is 4.
        list_dir (list): List of folder names in the data directory.
        data (list): Expected subdirectories in the data directory: ['train', 'val', 'test'].
    """
    def __init__(
            self,
            data_dir: str, 
            batch_size: int = 32, 
            num_workers: int = 4) -> None:
        """
        Initializes the CreateDataLoader class with directory and data loading parameters.

        Args:
            data_dir (str): Root directory path containing dataset subfolders.
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 32.
            num_workers (int, optional): Number of parallel data loading workers. Defaults to 4.
        """
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

        self.list_dir = os.listdir(self.data_dir)
        self.data = ['train', 'val', 'test']
    
    def _list_imagenes(self) -> list[str]:
        """
        Collects paths of all images in the 'train' directory.

        Returns:
            list[str]: A list of file paths for images in the 'train' folder.

        Raises:
            ValueError: If the 'train' folder does not exist in the dataset directory.
        """
        list_path_imagen = []

        if 'train' in self.list_dir:
            path_test = os.path.join(self.data_dir, 'train')
            list_folder = os.listdir(path=path_test)

            for folder in tqdm(list_folder, desc="read_folder"):
                path_labels = os.path.join(path_test, folder)
                list_images = os.listdir(path=path_labels)

                for image in tqdm(list_images, desc=f"read_folder_{folder}"):
                    condiction_image = any(
                    [image.endswith('.jpg'), 
                    image.endswith('.png')])

                    if condiction_image:
                        image_path = os.path.join(path_labels, image)
                        list_path_imagen.append(image_path)

        else:
            raise ValueError(
                """The 'train' folder is not present in the directory.
                Please ensure that the directory contains a 'train' folder.
                """)

        return list_path_imagen
    
    def calculate_mean_std(self) -> tuple[list[float], list[float]]:
        """
        Calculates the per-channel mean and standard deviation of all images in the 'train' folder.

        Returns:
            tuple[list[float], list[float]]: Mean and standard deviation for each color channel (RGB).

        Raises:
            ValueError: If no images are found in the 'train' folder.
        """
        paths_image = self._list_imagenes()
        
        if not paths_image:
            raise ValueError("No images found in the 'train' folder.")

        suma = np.zeros(3, dtype=np.float64)
        suma_squared = np.zeros(3, dtype=np.float64)
        pixeles_count = 0

        for path in tqdm(paths_image, desc="calculate_mean_std"):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.float32) / 255.0

            h, w, c = img_rgb.shape
            pixeles_count += h * w

            suma += np.sum(img_rgb, axis=(0, 1))
            suma_squared += np.sum(img_rgb ** 2, axis=(0, 1))

        mean = suma / pixeles_count

        var = (suma_squared / pixeles_count) - (mean ** 2)
        std = np.sqrt(var)

        return mean, std

    def datoloader_generar(self) -> list[DataLoader]:
        """
        Generates DataLoaders for each of the dataset folders: 'train', 'val', and 'test'.
        Applies normalization using computed mean and std from the training set.

        Returns:
            list[DataLoader]: A list containing DataLoaders for train, val, and test datasets.
        """
        list_batch = []
        means, std = self.calculate_mean_std()

        for folder in self.list_dir:

            if folder in self.data:
                root = os.path.join(self.data_dir, folder)
                transforms_images = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=means, std=std)
                ])

                datasets_images = datasets.ImageFolder(
                    root=root,
                    transform=transforms_images
                )

                dataloader_images = DataLoader(
                    datasets_images,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True
                )

                list_batch.append(dataloader_images)

        return list_batch
