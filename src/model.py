import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size // 2)
        self.hidden_layer2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
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
        metadata = {key: value for key, value in checkpoint.items()
                    if key != 'model_state_dict'}
        return metadata


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        game_overs: torch.Tensor
    ):
        """
        Performs a training step on a batch or a single experience.
        """

        predictions: torch.Tensor = self.model(states)

        if actions.dim() == 1:
            actions = actions.unsqueeze(1)

        q_value = predictions.gather(1, actions).squeeze(1)

        with torch.no_grad():

            next_predictions = self.model(next_states)
            next_action = torch.max(next_predictions, dim=1)[0]
            new_q_value = rewards + (self.gamma * next_action * (~game_overs))

        self.optimizer.zero_grad()
        loss = self.criterion.forward(q_value, new_q_value)
        loss.backward()
        self.optimizer.step()
