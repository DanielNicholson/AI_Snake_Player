import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os 

class Linear_Qnet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(file_name="model.pth"):
        model_folder_path = './model'

        file_name = os.path.join(model_folder_path, file_name)

        model = torch.load(PATH)
        model.eval()
        return model

        
        

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # trains the model based on the state of the game, the action taken, and the reward received
    def train_step(self, state, action, reward, next_state, done ):
        state      = torch.tensor(state, dtype = torch.float)
        next_state = torch.tensor(next_state, dtype = torch.float)
        reward     = torch.tensor(reward, dtype = torch.float)
        action     = torch.tensor(action, dtype = torch.float)

        if len(state.shape) == 1:
            # state is a single value
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            done = (done, )
        
        # 1: predicuted next action with current state, 
        pred = self.model(state)

        target = pred.clone()


        # calculate the new Q value for the next state if the game is not over to predict the reward
        for inx in range(len(done)):
            Q_new = reward[inx]

            # 2: Q_new = r + y * max(next_predicted Q value)
            if not done[inx]:
                Q_new = reward[inx] + self.gamma * torch.max(self.model(next_state[inx]))
            
            target[inx][torch.argmax(action).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        
