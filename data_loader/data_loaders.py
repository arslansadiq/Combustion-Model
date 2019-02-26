from base import BaseDataLoader
from data_loader.combustion_data import combustion_data
       
       
class CombustiontDataLoader(BaseDataLoader):
    
    #Combustion data loading demo using BaseDataLoader
    
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        
        '''
        Combustion data loading demo using BaseDataLoader
        '''
        
        self.data_dir = data_dir
        self.dataset = combustion_data(self.data_dir, train = training, process = True)
        
        super(CombustiontDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)