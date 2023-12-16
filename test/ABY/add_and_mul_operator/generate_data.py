import pandas as pd
import numpy as np
import os


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# 准备一份数据，1000行1列的数据，从1到1000
data = np.arange(1, 1001).reshape(1000, 1)


data.astype(np.int32)

data = pd.DataFrame(data, columns=['x'])

# id列名id，插入到第0列
data.insert(0, 'id', range(0, 0 + len(data)))

# 将数据保存到文件中
data.to_csv(os.path.join(data_path, "vector_add_and_mul_test_guest.csv"), index=False)
data.to_csv(os.path.join(data_path, "vector_add_and_mul_test_host.csv"), index=False)

