#The other main factor of RL model
# Helps the actor modify its action by objective evaluation, not follows the policy of the actor  
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the embedding + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_embedding = Embedding(static_size, hidden_size)
        self.dynamic_embedding = Embedding(dynamic_size, hidden_size)

        # Define the embedding & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_embedding(static)
        dynamic_hidden = self.dynamic_embedding(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the embedding + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super(Critic, self).__init__()

        # Define the embedding & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):

        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output
