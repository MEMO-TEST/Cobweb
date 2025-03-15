import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from tqdm import tqdm

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class MLPClassifier:
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate, lr=0.001, class_weights=None, weight_decay = 1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_dim, hidden_dims, output_dim, dropout_rate, weight_decay).to(self.device, dtype=torch.float32)
        # self.criterion = nn.CrossEntropyLoss()
        self.output_dim = output_dim
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            # self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=weight_decay)
        # self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)

    def train(self, train_loader, num_epochs, save_path, val_loader=None, retrain = False):
        if (not os.path.exists(save_path)) or retrain:
            best_val_accuracy = 0.0
            for epoch in range(num_epochs):
                self.model.train()  # 设置模型为训练模式
                running_loss = 0.0
                correct = 0
                total = 0
                total_step = len(train_loader)
                with tqdm(total=total_step, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                    for inputs, labels in train_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(inputs)

                        #计算loss
                        loss = self.criterion(outputs, labels)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        running_loss += loss.item()
                        pbar.set_postfix({"Loss": loss.item()})
                        pbar.update()

                        #将one-hot编码转化为标签数字形式
                        labels = torch.argmax(labels, dim = 1)
                        predicted = torch.argmax(outputs.data, dim = 1)
                        # predicted = outputs.data
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                epoch_loss = running_loss / total_step
                accuracy = correct / total

                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

                # self.scheduler.step()#更新学习率

                if(val_loader != None):
                    # Validation loop
                    self.model.eval()  # 设置为评估模式
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for val_inputs, val_labels in val_loader:
                            val_inputs = val_inputs.to(self.device)
                            val_labels = val_labels.to(self.device)
                            val_labels = torch.argmax(val_labels, dim=1)

                            val_outputs = self.model(val_inputs)
                            val_predicted = torch.argmax(val_outputs.data, dim=1)
                            val_total += val_labels.size(0)
                            val_correct += (val_predicted == val_labels).sum().item()

                    val_accuracy = val_correct / val_total
                    print(f"Epoch [{epoch+1}/{num_epochs}] (Val), Accuracy: {val_accuracy:.4f}")

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        self.save_model(save_path)
                else:
                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy
                        self.save_model(save_path)
            self.model.eval()
        else:
            self.load_model(save_path)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model parameters saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model parameters loaded from {path}")

    def test(self, test_loader):
        self.model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        true_positives = 0
        predicted_positives = 0
        actual_positives = 0
        true_positives = [0] * self.output_dim
        predicted_positives = [0] * self.output_dim
        actual_positives = [0] * self.output_dim
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)

                labels = torch.argmax(labels, dim=1)
                predicted = torch.argmax(outputs.data, dim = 1)
                # predicted = outputs.data
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for index in range(self.output_dim):
                    true_positives[index] += ((predicted == index) & (labels == index)).sum().item()
                    predicted_positives[index] += (predicted == index).sum().item()
                    actual_positives[index] += (labels == index).sum().item()

        accuracy = correct / total
        precision = [0] * self.output_dim
        recall = [0] * self.output_dim
        f1_score = [0] * self.output_dim
        for index in range(self.output_dim):
            precision[index] = (true_positives[index] / predicted_positives[index])
            recall[index] = (true_positives[index] / actual_positives[index])
            f1_score[index] = (2 * (precision[index] * recall[index]) / (precision[index] + recall[index]))
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1_score}")

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            if type(x) is np.ndarray:
                x = torch.from_numpy(x)
            x = x.to(self.device, dtype=torch.float32)
            outputs = self.model(x)
            # predicted = torch.argmax(outputs.data, dim = 1)
            if(x.dim() > 1):
                predicted = torch.argmax(outputs.data, dim = 1)
            else:
                predicted = torch.argmax(outputs.data, dim = 0)
        return predicted

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate, weight_decay):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay  # L2 正则化参数

        for i in range(len(dims) - 1):
            # layer = nn.Linear(dims[i], dims[i + 1])
            # nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # 使用 He 初始化
            # nn.init.zeros_(layer.bias)
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout_rate))
            
            # if i < len(dims) - 2:
            #     self.layers.append(nn.LeakyReLU())
            # else:
            #     self.layers.append(nn.Softmax(dim=1))
            # self.layers.append(nn.Dropout(self.dropout_rate))
        # self.layers = self.layers[:-1]  # 移除最后一个 Dropout 层

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        # out = nn.functional.softmax(out, dim=1)
        # out = torch.sigmoid(out)
        return out


if __name__ == 'main':
    # 创建 MLPClassifier 对象
    classifier = MLPClassifier(input_dim=10, hidden_dims=[64, 32, 16, 8, 4], output_dim=2)

    # 定义训练数据和标签
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))

    # 训练模型
    classifier.train(x_train, y_train, num_epochs=100, save_path='../model_info/model.pt')

    # 定义测试数据
    x_test = torch.randn(10, 10)

    # 进行预测
    predictions = classifier.predict(x_test)
