import os
import torch
data_dir_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "data")
print(data_dir_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)