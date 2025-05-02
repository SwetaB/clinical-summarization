import os
import pandas as pd
from datasets import Dataset, load_dataset, load_from_disk


class DataLoading:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        if os.path.isfile(self.data_path):
            if self.data_path.lower().endswith('.csv'):
                df = pd.read_csv(self.data_path)
                self.dataset = Dataset.from_pandas(df)
            else:
                print("please load the file with .csv extension")
        elif os.path.isdir(self.data_path):
            self.dataset = load_from_disk(self.data_path)
        else:
            # maybe it's a HF dataset name
            self.dataset = load_dataset(self.data_path)
        return self.dataset
        