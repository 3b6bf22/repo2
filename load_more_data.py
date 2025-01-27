import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import numpy as np
import pandas
from zipfile import ZipFile

UCI_DATASET_URLS = {
    'protein': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',
    'ct': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip',
    'workloads': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00493/datasets.zip',
    'msd': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
}

UCI_DATASET_DATAFILE = {
    'protein': 'CASP.csv',
    'ct': 'slice_localization_data.csv',
    'workloads': 'Range-Queries-Aggregates.csv',
    'msd': 'YearPredictionMSD.txt',
}

class UCIDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def download_and_load_uci_dataset(name):
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    scaler = StandardScaler()
    if name == 'adult':
        if not os.path.exists(os.path.join(data_dir, 'adult.csv')):
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            columns = [
                'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country', 'income'
            ]
            df = pd.read_csv(url, names=columns, engine='python')
            df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'object' else x)
            df_encoded = pd.get_dummies(df, drop_first=True)
            numerical_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
            df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
            bool_columns = df_encoded.select_dtypes(include=["bool"]).columns
            df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
            
            df = df_encoded
            df.to_csv(os.path.join(data_dir, 'adult.csv'), index=False)
        else:
            df = pd.read_csv(os.path.join(data_dir, 'adult.csv'))
        X = df.drop('income_ >50K', axis=1).values/10  # 'income_>50K' 是目标列
        y = df['income_ >50K'].values
    elif name in UCI_DATASET_URLS.keys():
        file_dir = os.path.join(data_dir, name)
        if not os.path.exists(file_dir):
            print(f"creating directory {file_dir}")
            os.makedirs(file_dir)

        url = UCI_DATASET_URLS[name]
        data_file = os.path.join(file_dir, UCI_DATASET_DATAFILE[name])
        if not os.path.exists(data_file):
            download_file = os.path.split(url)[-1].replace('%20', ' ')
            if not os.path.exists(os.path.join(file_dir, download_file)):
                print(f"downloading file...")
                os.system(f"wget {url} -P {file_dir}/")

            if '.zip' == download_file[-4:]:
                with ZipFile(os.path.join(file_dir, download_file), 'r') as zip_obj:
                    zip_obj.printdir()
                    zip_obj.extractall(file_dir)

            if name == 'workloads':
                os.system(f"mv {file_dir}/Datasets/* {file_dir}/")
                os.system(f"rm -rf {file_dir}/Datasets")

        if name == 'msd':
            x = np.array(pandas.read_csv(data_file, header=None))[:, 1:]
        else:
            x = np.array(pandas.read_csv(data_file))

        if name in ['protein', 'msd']:
            y = x[:, 0]
            x = x[:, 1:]
        elif name in ['ct']:
            y = x[:, -1]
            x = x[:, 1:-1]
        elif name in ['workloads']:
            x = np.unique(x[:, 1:], axis=0)
            while (1):
                if not np.any(np.isnan(x)):
                    break
                idx = np.where(np.sum(np.isnan(x), axis=1) == 1)[0][0]
                x = np.delete(x, idx, 0)
            y = x[:, -1]
            x = x[:, :-1]
        else:
            y = x[:, -1]
            x = x[:, :-1]

        X = scaler.fit_transform(x)

        if name=='protein':
            X = X/10.0
            y = y/10.0
        elif name=='ct':
            X = X/45.0
            y = y/10.0
        elif name=='workloads':
            X = X/7.5
            y = y/100.0
        elif name=='msd':
            X = X/20.0
            y = y/10.0
            
        y = y
    else:
        raise ValueError("Dataset not found")

    return X, y

def get_uci_loaders(name):
    X, y = download_and_load_uci_dataset(name)
    return X, y