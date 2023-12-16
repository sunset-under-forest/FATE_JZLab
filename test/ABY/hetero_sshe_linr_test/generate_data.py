import pandas as pd
import numpy as np
import os


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# 准备一份线性回归的数据，三个特征，一个标签，1000个样本
# Y = X1 + 2 * X2 + 3 * X3 + 4
X1 = np.arange(2001, 3001).reshape(1000, 1)
X2 = np.arange(1001, 2001).reshape(1000, 1)
X3 = np.arange(1, 1001).reshape(1000, 1)
Y = X1 + 2 * X2 + 3 * X3 + 4

# Y, X1, X2, X3)
data = np.hstack((Y, X1, X2, X3))

data.astype(np.int32)

data = pd.DataFrame(data, columns=['y', 'x1', 'x2', 'x3'])

# id列名id，插入到第0列
data.insert(0, 'id', range(0, 0 + len(data)))

# 纵向联邦
# 取id, y, x1列
data_guest = data[['id', 'y', 'x1']]
# 取id, x2, x3列
data_host = data[['id', 'x2', 'x3']]

# 将数据保存到文件中
data_guest.to_csv(os.path.join(data_path, "hetero_sshe_linr_test_guest.csv"), index=False)
data_host.to_csv(os.path.join(data_path, "hetero_sshe_linr_test_host.csv"), index=False)

