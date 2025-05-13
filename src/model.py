import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x
    
    def save(self, file_name, metadata=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
        }
        if metadata:
            checkpoint.update(metadata)
        torch.save(checkpoint, file_name)

    def load(self, file_name):
        checkpoint = torch.load(file_name, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        metadata = {key: value for key, value in checkpoint.items() if key != 'model_state_dict'}
        return metadata

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            action_index = torch.argmax(action).item()

            if action_index >= target.shape[1]:
                print(f"⚠️ Warning: action_index {action_index} is out of bounds! Resetting to 0.")
                action_index = 0 

            target[idx][action_index] = Q_new
  
        self.optimizer.zero_grad()
        loss = self.criterion.forward(target, pred)
        loss.backward()
        self.optimizer.step()
