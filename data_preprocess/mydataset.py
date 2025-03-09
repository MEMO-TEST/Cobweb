from torch.utils.data import Dataset, DataLoader
import numpy as np


# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x = self.data[index].astype(np.float32)
        y = self.labels[index].astype(np.float32)
        return x, y

    def __len__(self):
        return len(self.data)
    
# 创建训练集和测试集的数据加载器
def create_data_loaders(train_data, train_labels, test_data, test_labels, batch_size):
    train_dataset = MyDataset(train_data, train_labels)
    test_dataset = MyDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
