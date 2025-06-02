import os

import cv2
import numpy as np
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm



class CreateDataLoader:
    def __init__(
            self,
            data_dir: str, 
            batch_size: int = 32, 
            num_workers: int = 4) -> None:

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers

        self.list_dir = os.listdir(self.data_dir)
        self.data = ['train', 'val', 'test']
    
    def _list_imagenes(self) -> list[str]:
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
