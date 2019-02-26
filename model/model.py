import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import json
import argparse


class CombustionModel(BaseModel):
    def __init__(self, num_features=7):
        super(CombustionModel, self).__init__()
        sizes = self.get_botleneck_size()              #sizes for bottlenecks
        self.Fc1 = nn.Linear(in_features = 2, out_features = 500, bias=True)
        self.Fc2 = nn.Linear(in_features = 500, out_features = 500, bias=True)
        self.Fc3_bottleneck = nn.Linear(in_features = 500, out_features = sizes[0], bias=True)
        self.Fc4 = nn.Linear(in_features = sizes[0], out_features = 500, bias=True)
        self.Fc5_bottleneck = nn.Linear(in_features = 500, out_features = sizes[1], bias=True)
        self.Fc6 = nn.Linear(in_features = sizes[1], out_features = 500, bias=True)
        self.Fc7_bottleneck = nn.Linear(in_features = 500, out_features = sizes[2], bias=True)
        self.Fc8 = nn.Linear(in_features = sizes[2], out_features = 500, bias=True)
        self.Fc9_bottleneck = nn.Linear(in_features = 500, out_features = sizes[3], bias=True)
        self.Fc10 = nn.Linear(in_features = sizes[3], out_features = 500, bias=True)
        self.Fc11_bottleneck = nn.Linear(in_features = 500, out_features = sizes[4], bias=True)
        self.Fc12 = nn.Linear(in_features = sizes[4], out_features = num_features, bias=True)
        
    def get_botleneck_size(self):
         parser = argparse.ArgumentParser(description='BottleNeck')
         parser.add_argument('-c', '--config', default='config.json', type=str,
                           help='config file path (default: None)')
         args = parser.parse_args()
         config = json.load(open(args.config))
         bottleneck_size = config['arch']['bottleneck_size']
         if type(bottleneck_size) is list:
             if len(bottleneck_size) == 5:    #comparing it to 5 because we have 5 bottlenecks in the model
                 pass
             else:
                 raise Exception("bottleneck's list length in config.json file is not equal to number of bottnecks in model's structure")
             return bottleneck_size
         elif type(bottleneck_size) is int:
             list_tmp = []
             for i in range(5):
                 list_tmp.append(bottleneck_size)
             bottleneck_size = list_tmp
             del(list_tmp)
             return bottleneck_size
         
    def forward(self, x):
        '''
        This function computes the network computations based on input x 
        built in the constructor of the the CombustionModel
        '''
        
        '''First Layer'''
        x = self.Fc1(x)
        x = F.relu(x)
        
        '''First ResNet Block'''
        res_calc = self.Fc2(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc3_bottleneck(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Second ResNet Block'''
        res_calc = self.Fc4(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc5_bottleneck(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Third ResNet Block'''
        res_calc = self.Fc6(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc7_bottleneck(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Fourth ResNet Block'''
        res_calc = self.Fc8(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc9_bottleneck(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Fifth ResNet Block'''
        res_calc = self.Fc10(x)
        res_calc = F.relu(res_calc)
        res_calc = self.Fc11_bottleneck(res_calc)
        x = F.relu(torch.add(x, res_calc))
        
        '''Regression layer'''
        return self.Fc12(x)
        
