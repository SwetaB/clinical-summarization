import os
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split


class DatasetManager():
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        if os.path.isfile(self.data_path):
            if self.data_path.lower().endswith('.csv'):
                self.dataset = pd.read_csv('csv', data_files=self.data_path)
        elif os.path.isdir(self.data_path):
            self.dataset = load_from_disk(self.data_path)
        else:
            # maybe it's a HF dataset name
            self.dataset = load_dataset(self.data_path)
        


