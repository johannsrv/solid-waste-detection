import os
import numpy as np
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_np, label = self.data[idx]
        image = Image.fromarray(image_np.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, label

class ProcessingData:
    def __init__(self, data):
        self.data = data

    def save_processed_data(self, file_path: str = 'data/processed'):
        os.makedirs(file_path, exist_ok=True)
        self._transform()

        for idx, (image, label) in enumerate(self.data):
            image = Image.fromarray(image.astype(np.uint8))

            if self.transform:
                image = self.transform(image)
                if isinstance(image, torch.Tensor):
                    image = to_pil_image(image)

            label_dir = os.path.join(file_path, str(label))
            os.makedirs(label_dir, exist_ok=True)
            image.save(os.path.join(label_dir, f'image_{idx}.png'))

    def _calcualte_mean_std(self):
        channel_sums = np.zeros(3)
        channel_squared_sums = np.zeros(3)
        num_image = 0

        for image_np, _ in self.data:
            image = image_np.astype(np.float32) / 255.0

            channel_sums += np.sum(image, axis=(0, 1))
            channel_squared_sums += np.sum(image ** 2, axis=(0, 1))
            num_image +=1

        self.mean = (
            channel_sums 
            / (num_image * image_np.shape[0] 
            * image_np.shape[1]))
        
        self.std = np.sqrt(
            channel_squared_sums 
            / (num_image * image_np.shape[0] * image_np.shape[1])
            - self.mean ** 2)
        
        print(f"Calculated mean: {self.mean}, std: {self.std}")

    def _transform(self):
        self._calcualte_mean_std()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            # transforms.GaussianBlur(kernel_size=5),
            transforms.RandomRotation(30)  
        ])
    
    def split_data(self, train_ratio=0.8) -> Tuple[Subset, Subset, Subset]:
        self._transform()
        dataset = CustomImageDataset(self.data, transform=self.transform)
        
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size]
            )
        
        return train_dataset, val_dataset, test_dataset
    
    def data_loader(
        self, 
        train_dataset: Subset, 
        val_dataset: Subset, 
        test_dataset: Subset, 
        batch_size=32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        dataLoader_train = DataLoader(
            train_dataset,
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4)
        
        dataLoader_val = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4)
        
        dataLoader_test = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4)
        
        return dataLoader_train, dataLoader_val, dataLoader_test