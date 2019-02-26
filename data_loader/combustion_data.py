import torch.utils.data as data
from sklearn import preprocessing
from sklearn.utils import shuffle
#import numpy as np
import errno
import torch
import os
import pandas as pd


class combustion_data(data.Dataset):
    
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    
    def __init__(self, root, train=True, transform=None, target_transform=None, process=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        
        
        if process:        
            self.processing()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + 'Put it there...!')
            
        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file)))
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input, target) where target is index of the target class.
        """
        if self.train:
            inp, target = self.train_data[index], self.train_labels[index]
        else:
            inp, target = self.test_data[index], self.test_labels[index]
           
        return inp, target
    
    def processing(self):
        
        if self._check_exists():
            return
        
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        print("Processing Data...")    
        df = pd.DataFrame()
        raw_data_dir = os.path.join(self.root, self.raw_folder) + '/'
        for fn in os.listdir(raw_data_dir):
            tmp = pd.read_csv(os.path.join(raw_data_dir, fn))
            tmp['p'] = int(fn.replace('bar.csv', ''))
            df = df.append(tmp)
            
        ### shuffling and splitting of data into training and test data
        df = shuffle(df)
        train_d = df[:int(df.shape[0]*0.8)]
        test_d = df[int(df.shape[0]*0.8):]
        
        ### training data
        train_in_data = train_d[['p', 'he']]
        train_out_data = train_d[['rho','T','thermo:mu','Cp','thermo:psi','thermo:alpha','thermo:as']]
        in_scalar1 = preprocessing.StandardScaler()
        train_in_data = in_scalar1.fit_transform(train_in_data)
        train_in_data = torch.from_numpy(train_in_data)         #training input
        out_scalar1 = preprocessing.StandardScaler()
        train_out_data = out_scalar1.fit_transform(train_out_data)
        train_out_data = torch.from_numpy(train_out_data)       #training output
        
        ### testing data
        test_in_data = test_d[['p', 'he']]
        test_out_data = test_d[['rho','T','thermo:mu','Cp','thermo:psi','thermo:alpha','thermo:as']]
        in_scalar2 = preprocessing.StandardScaler()
        test_in_data = in_scalar2.fit_transform(test_in_data)
        test_in_data = torch.from_numpy(test_in_data)         #testing input
        out_scalar2 = preprocessing.StandardScaler()
        test_out_data = out_scalar2.fit_transform(test_out_data)
        test_out_data = torch.from_numpy(test_out_data)      #testing output    
        
        train_in_data = train_in_data.float()
        train_out_data = train_out_data.float()
        test_in_data = test_in_data.float()
        test_out_data = test_out_data.float()
        training_set = (train_in_data, train_out_data)
        test_set = (test_in_data, test_out_data)
        
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)
            
        print("Done...")
        
        
