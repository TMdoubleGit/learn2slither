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
        done = torch.tensor(np.array(done), dtype=torch.bool)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
    
        if action.dim() == 2 and action.shape[1] > 1:
            action = torch.argmax(action, dim=1).unsqueeze(-1)

        # pred = self.model(state)
        # target = pred.clone()

        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
        #     action_index = torch.argmax(action).item()

        #     target[idx][action_index] = Q_new

        pred = self.model(state)
        target = pred.clone()

        next_predictions = self.model(next_state)

        next_action = torch.max(next_predictions, dim=1)[0]

        if isinstance(done, tuple) or isinstance(done, list):
            done = torch.tensor(done, dtype=torch.bool)
        elif isinstance(done, bool):
            done = torch.tensor([done], dtype=torch.bool)

        new_q_value = reward + self.gamma * next_action * (~done)

        if len(action.shape) == 1:
            action_indices = action.unsqueeze(1)
        else:
            action_indices = action

        batch, num_actions = pred.shape
        action_indices = torch.clamp(action_indices, min=0, max=num_actions - 1)

        # Vérifier les formes avant scatter
        # assert target.shape == (batch, num_actions), f"target shape: {target.shape}"
        # assert action_indices.shape == (batch, 1), f"action_indices shape: {action_indices.shape}"
        # assert new_q_value.unsqueeze(-1).shape == (batch, 1), f"new_q_value shape: {new_q_value.unsqueeze(-1).shape}"

        target.scatter_(1, action_indices, new_q_value.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss = self.criterion.forward(target, pred)
        loss.backward()
        self.optimizer.step()
