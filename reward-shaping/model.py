import torch
import torch.nn as nn
from gymnasium import spaces # For type hinting
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

HIDDEN_DIM_DEFAULT = 256

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = HIDDEN_DIM_DEFAULT):
        super(CustomCNN, self).__init__(observation_space, features_dim)

        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("CustomCNN expects a Dict observation space.")
        if 'grid' not in observation_space.spaces:
            raise ValueError("The Dict observation space must contain 'grid', 'sign_array', and 'last_action' keys.")

        grid_space = observation_space.spaces['grid']

        n_input_channels = grid_space.shape[0]

        self.cnn_layer1 = nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn_layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cnn_layer3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy_grid_input = torch.as_tensor(grid_space.sample()[None]).float()
            x = self.relu1(self.cnn_layer1(dummy_grid_input))
            x = self.pool1(x)
            x = self.relu2(self.cnn_layer2(x))
            x = self.pool2(x)
            x = self.relu3(self.cnn_layer3(x))
            n_flatten_cnn = self.flatten(x).shape[1]

        n_flatten = n_flatten_cnn 

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        grid_input = observations['grid']

        x = self.relu1(self.cnn_layer1(grid_input))
        x = self.pool1(x)
        x = self.relu2(self.cnn_layer2(x))
        x = self.pool2(x)
        x = self.relu3(self.cnn_layer3(x))
        x_cnn_flattened = self.flatten(x)


        return self.linear(x_cnn_flattened)