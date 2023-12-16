from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os
import pandas as pd
from const import data_dir_path, device

data_name = "OrganAMNIST.csv"
data_guest_name = "OrganAMNIST_guest.csv"
data_host_name = "OrganAMNIST_host.csv"

data_path = os.path.join(data_dir_path, data_name)
data_guest_path = os.path.join(data_dir_path, data_guest_name)
data_host_path = os.path.join(data_dir_path, data_host_name)

data = pd.read_csv(data_path)
data_guest = pd.read_csv(data_guest_path)
data_host = pd.read_csv(data_host_path)

feature_names = ['x' + str(i) for i in range(data_guest.shape[1] - 2)]

feature = data[feature_names]
feature_guest = data_guest[feature_names]
feature_host = data_host[feature_names]

label = data['y']
label_guest = data_guest['y']
label_host = data_host['y']

# 转化成tensor
feature = torch.from_numpy(feature.values).float()
feature_guest = torch.from_numpy(feature_guest.values).float()
feature_host = torch.from_numpy(feature_host.values).float()

label = torch.from_numpy(label.values).long()
label_guest = torch.from_numpy(label_guest.values).long()
label_host = torch.from_numpy(label_host.values).long()

print(feature.shape)
print(feature_guest.shape)
print(feature_host.shape)

from psg import run

# 超参数

batch_size = 32
epochs = 20
lr = 2e-4
classes = 11
channels = 1
img_size = 64
latent_dim = 100
log_interval = 100
seed = 42

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

from torch.utils.data import Dataset


class MyDateset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        return transform(self.features[item]).reshape(1, img_size, img_size), self.labels[item]

    def __len__(self):
        return self.features.shape[0]


feature = feature.reshape(-1, 1, 28, 28)
feature_guest = feature_guest.reshape(-1, 1, 28, 28)
feature_host = feature_host.reshape(-1, 1, 28, 28)

# reshape
train_dataset = MyDateset(feature, label)
train_dataset_guest = MyDateset(feature_guest, label_guest)
train_dataset_host = MyDateset(feature_host, label_host)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_data_loader_guest = DataLoader(dataset=train_dataset_guest, batch_size=batch_size, shuffle=True)
train_data_loader_host = DataLoader(dataset=train_dataset_host, batch_size=batch_size, shuffle=True)

generator = run(train_data_loader, batch_size, epochs, lr, classes, channels, img_size, latent_dim)
generator_guest = run(train_data_loader_guest, batch_size, epochs, lr, classes, channels, img_size, latent_dim)
generator_host = run(train_data_loader_host, batch_size, epochs, lr, classes, channels, img_size, latent_dim)

# 每个样本都生成一个假样本
img_list = []
img_list_guest = []
img_list_host = []

viz_z = torch.zeros((batch_size, latent_dim), device=device)
viz_noise = torch.randn(batch_size, latent_dim, device=device)  # 噪声
nrows = batch_size // 8
viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(8)])).to(device)  # 虚拟标签
import torchvision.utils as vutils

with torch.no_grad():
    viz_sample = generator(viz_noise, viz_label)
    viz_sample_guest = generator_guest(viz_noise, viz_label)
    viz_sample_host = generator_host(viz_noise, viz_label)
    img_list.append(vutils.make_grid(viz_sample, padding=2, normalize=True))
    img_list_guest.append(vutils.make_grid(viz_sample_guest, padding=2, normalize=True))
    img_list_host.append(vutils.make_grid(viz_sample_host, padding=2, normalize=True))

# 数据分发

# 分发比例

distribution = [0.5, 0.5]

fake_sample_guest = []
fake_sample_host = []

for i in range(len(img_list)):
    fake_sample_guest.append((img_list_guest[i] * distribution[0], viz_label[i]))
    fake_sample_host.append((img_list_guest[i] * distribution[0], viz_label[i]))

# 保存

torch.save(fake_sample_guest, "fake_sample_guest.pt")
torch.save(fake_sample_host, "fake_sample_host.pt")
