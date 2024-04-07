import numpy as np
import minari
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "seed": torch.Tensor([x.seed for x in batch]),
        "total_steps": torch.Tensor([x.total_timesteps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.observations["observation"]) for x in batch],
            batch_first=True,
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions) for x in batch], batch_first=True
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.tensor(x).float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def make_offline_rl_dataset():
    dataset = minari.load_dataset("antmaze-umaze-v0", download=True)
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
    )
    env = dataset.recover_environment()
    return dataloader, env


if __name__ == "__main__":
    num_epochs = 3
    dataloader, env = make_offline_rl_dataset()

    observation_space = env.observation_space["observation"]
    action_space = env.action_space
    policy_net = PolicyNetwork(np.prod(observation_space.shape), action_space.shape[0])
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in dataloader:
            a_pred = policy_net(batch["observations"][:, :-1])
            loss = loss_fn(a_pred, a_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

    state = env.reset()[0]
    te, tr = False, False
    r = 0
    while not (te or tr):
        action = policy_net(state["observation"])
        state, reward, te, tr, _ = env.step(action.detach().numpy())
        r += reward
    print(f"Total online evaluation reward: {reward}")
    env.close()
