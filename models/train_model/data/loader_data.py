import os
import random

import cv2
import numpy as np

class LoaderData:

    def __init__(self, folder_path: str) -> None:
        random.seed(14)

        self.name_classes = {}
        self.data = []

        self.folder_path = folder_path

    def _read_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Image at {image_path} could not be read.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def loader_data(self) -> None:
        extension = ['.jpg', '.jpeg', '.png']
        count_classes = 0
        class_date = os.listdir(self.folder_path)

        for class_name in class_date:
            self.name_classes[class_name] = [count_classes]
            class_path = os.path.join(self.folder_path, class_name)
            count_classes += 1

            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                for image in images:
                    image_path = os.path.join(class_path, image)
                    image_extension = os.path.splitext(image)[1].lower()
                    if image_extension in extension:
                        image = self._read_image(image_path)
                        self.data.append((image, self.name_classes[class_name][0]))

        random.shuffle(self.data)
        return self.name_classes, self.data

    
    