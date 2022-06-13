import torch
import random
import numpy as np

import os 

from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_Qnet, QTrainer

from helper import plot 

MAX_MEM = 100_000
BATCH_SIZE = 1000
Learning_Rate = 0.001

class Agent:

    def __init__(self):

        file_name="model.pth"
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        self.no_game = 0
        self.epsilon = 0  
        self.gamma   = 0.8 
        self.memory  = deque(maxlen=MAX_MEM)
        self.model   = Linear_Qnet(16, 512, 3)
        
        self.trainer = QTrainer(self.model, lr=Learning_Rate, gamma=self.gamma)

       

    

    # gets the current state of the game which contains 16 values: 6 tell us if there is any danger, 4 tell us the current direction of the snake, and 4 tell us the direction to the food
    def get_state(self, game):

        head = game.snake[0]

        # points immediately surrounding the snake head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # points 2 block away from the snake head
        point_l_2 = Point(head.x - 40, head.y)
        point_r_2 = Point(head.x + 40, head.y)
        point_u_2 = Point(head.x, head.y - 40)
        point_d_2 = Point(head.x, head.y + 40)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            len(game.snake),

            game.score,

            # 3 values tell us if there is danger of collision immediately
            # (0: danger forward, 1: danger right, 2: danger left)

            # straigh ahead
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # left
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # 3 values tell us if there is danger of collision 2 blocks away
            # straigh ahead
            (dir_r and game.is_collision(point_r_2)) or 
            (dir_l and game.is_collision(point_l_2)) or
            (dir_u and game.is_collision(point_u_2)) or
            (dir_d and game.is_collision(point_d_2)),

            # right
            (dir_u and game.is_collision(point_r_2)) or
            (dir_d and game.is_collision(point_l_2)) or 
            (dir_l and game.is_collision(point_u_2)) or
            (dir_r and game.is_collision(point_d_2)),

            # left
            (dir_u and game.is_collision(point_l_2)) or
            (dir_d and game.is_collision(point_r_2)) or
            (dir_r and game.is_collision(point_u_2)) or
            (dir_l and game.is_collision(point_d_2)),

            # 4 values specify the direction of the snake 
            # (0: left, 1: right, 2: up, 3: down)
            # movement direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # 4 values specify the position of the food
            # (0: left, 1: right, 2: up, 3: down)
            # food location
            game.food.x < game.head.x, # food is left
            game.food.x > game.head.x, # food is right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y, # food is down
            ]

        return np.array(state, dtype=int)

    # stores the state, action, reward, and next state in the memory. Used to train the model on batches of old moves after each game
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

        # if memory is full, remove the oldest one
        if len(self.memory) > MAX_MEM:
            self.memory.popleft()

    # after each game, train the model on the memory
    def train_long_memory(self):

        # if there are enough moves in memory to train on, take a random batch of moves and train on them
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)

        # if there are not enough moves in memory to train on, take the entire memory and train on it
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
           

    # used to train the memory after each move
    def train_short_memory(self, state, action, reward, next_state,  done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation

        # the more games played, the more epsilon is decreased, the less random the moves will be.
        self.epsilon = 80 - self.no_game

        final_move = [0,0,0]

        # if the random number is less than epsilon, make a random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1

        # as more games have been played, the model is more likely to be correct, use model to make a move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1

        return final_move

    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state of the game
        old_state = agent.get_state(game)

        # get move based on current state (Will be random moves at first, then will be model moves)
        final_move = agent.get_action(old_state)

        # apply move and get new state
        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        # train short term memory on the move just made
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # remember the move just made
        agent.remember(old_state, final_move, reward, new_state, done)
        
        # if the game is over
        if done:

            # reset game
            game.reset()
            agent.no_game += 1

            # train long term memory
            agent.train_long_memory()

            # if the game is over we have a new high score, save the model
            if score > record:
                record = score
                agent.model.save()

            print("Game: ", agent.no_game, " Score: ", score, " High Score: ", record)


            # plot the scores and mean score onto the graph
            total_score += score
            mean_score = total_score / agent.no_game

            plot_scores.append(score)
            plot_mean_scores.append(mean_score)    

            plot(plot_scores, plot_mean_scores)

            

            

if __name__ == '__main__':
    train()