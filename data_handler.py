import os
import kaggle
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataHandler:
    def __init__(self, dataset_name, img_size=(224, 224), batch_size=32):
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.dataset_dir = Path('datasets')
        
    def download_dataset(self):
        """Download the dataset from Kaggle"""
        self.dataset_dir.mkdir(exist_ok=True)
        
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.dataset_dir,
                unzip=True
            )
            print(f"Dataset successfully downloaded to {self.dataset_dir}")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def create_data_generator(self, subset_path, augment=False):
        """Create a data generator for training/validation"""
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            horizontal_flip=augment,
            rotation_range=20 if augment else 0,
            zoom_range=0.2 if augment else 0,
            width_shift_range=0.2 if augment else 0,
            height_shift_range=0.2 if augment else 0,
        )
        
        return datagen.flow_from_directory(
            subset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
    
    def get_class_names(self):
        """Get list of class names from the dataset directory"""
        train_dir = self.dataset_dir / 'train'  # Adjust path based on your dataset structure
        if train_dir.exists():
            return [d.name for d in train_dir.iterdir() if d.is_dir()]
        return []

    def print_dataset_info(self):
        """Print information about the dataset"""
        print("\nDataset Structure:")
        for item in self.dataset_dir.glob('**/*'):
            if item.is_dir():
                files = len(list(item.glob('*.*')))
                if files > 0:
                    print(f"- {item.relative_to(self.dataset_dir)}: {files} files")