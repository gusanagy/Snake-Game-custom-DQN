import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class RotationalActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class RotationalReLU(nn.Module):
    def __init__(self, angle=0.5):
        super(RotationalReLU, self).__init__()
        self.angle = angle

    def forward(self, x):
        return torch.max(x, x.mul(self.angle).cos().mul(x.sign()))
class RAFActivation(nn.Module):
    def __init__(self, p=2):
        super(RAFActivation, self).__init__()
        self.p = p

    def forward(self, x):
        return (1 / self.p) * torch.atan(self.p * x)
    
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.raf = RAFActivation(p=2)  # Usando a função de ativação RAF com p=2
        self.rot_activation = RotationalActivation()  # Camada com a "Rotational Activation Function"
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        #x = self.bn1(x)
        x = self.raf(self.fc2(x))  # Aplicando a RAF na segunda camada
        #x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        #x = self.rot_activation(x)  # Aplicar a "Rotational Activation Function"
        #x = RotationalReLU()(x) 
        x = self.fc3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1: 
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        #1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        #2: Q_new = r + y * max(next_pred Q value) -> only do this if not done

        #pred.clone()

        #preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

