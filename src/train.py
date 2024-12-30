from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import os

# Get the current working directory
current_path = os.getcwd()

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        self.path = current_path + "\\final_dqn.pt"
        torch.save(self.model.state_dict(), self.path)
        return

    def load(self):
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = current_path + "\\final_dqn.pt"
        self.model = DQN.to(device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return



DQN = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, env.action_space.n)
)



if __name__ == "__main__":
   pass


# I did all the training and experimatation on COLAB, the notebook is in the 
# the represirory and is called "Training_for_HIV_env.ipynb"