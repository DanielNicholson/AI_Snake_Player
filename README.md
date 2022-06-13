# AI Snake Player

## Project Overview
Using reinforcement learning to teach a nueral network how to play snake!

Project Steps:
1. Create a snake game
2. Create an agent to control the game
3. Create a model to learn how to play the game
4. Add in a human controlled game that can feed information to the model to train based on human interactions


## Code 
**File overview:**  
1. game.py       = The game of snake  
2. agent.py      = Agent that can generate a move and feed it into the game to play snake  
3. model.py      = A neural network with 16 input nodes and 3 output nodes  
4. game_human.py = A human controlled version of the snake game which can feed informaion into the model for training  


## Installation
To run the code locally you will need:  
- Python 3.7+
- Python Packages:
  - Pygame
  - NumPy
  - PyTorch

## Running the Code
To run the code, run the agent.py file to automatically train the model or run the game_human.py file to train the model with human inputs
