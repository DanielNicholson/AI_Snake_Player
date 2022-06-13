import pygame
import random
from enum import Enum
from collections import namedtuple

import numpy as np

import math

pygame.init()
font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# point is used to represent a position on the game board
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 200

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.reset()
        
        
    # used to reset the game to the start state after a game has finished.
    def reset(self):

        # initialist the snake
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-20, self.head.y),
                      Point(self.head.x-(2*20), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    
    # places a new food at a random location on the board
    def _place_food(self):
        x = random.randint(0, (self.w-20 )//20 )*20 
        y = random.randint(0, (self.h-20 )//20 )*20
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    # check to see if the snake have moved closer to the food, if so, return true
    def _moved_closer(self):
        dist_x = self.head.x - self.food.x
        dist_y = self.head.y - self.food.y

        old_dist_x = self.snake[1].x - self.food.x
        old_dist_y = self.snake[1].y - self.food.y

        dist = math.hypot(self.head.x-self.food.x, self.head.y - self.food.y)
        old_dist = math.hypot(self.snake[1].x - self.food.x, self.snake[1].y - self.food.y)

        return dist < old_dist

            
    # pass in an action from the agent and update the game state
    def play_step(self, action):
        self.frame_iteration += 1


        # 1. check to see if the user has quit.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check to see if the snake has found the food or collided with something
        reward = 0  # reward is given if the snake finds the food or negative reward if it collides with something
        game_over = False

        # check to see if the snake has collided with something or hasn't found the food for too many turns.
        if self.is_collision() or self.frame_iteration > (100 * len(self.snake)):
            game_over = True
            
            # if snake has collided with itself, give a big negative reward
            if self.collision_with_self():
                reward = -20
            # else the snak has collided with a boundary, give a small negative reward
            else:
                reward = -10    
            return reward, game_over, self.score
            
        # 4. check to see if the snak has found the food and place new food if needed
        if self.head == self.food:
            self.score += 1
            reward = 10 
            self._place_food()
        else:
            self.snake.pop() # if the snake did not find the food, then remove the last piece of the snake so it moves forward

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return reward, game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - 20 or pt.x < 0 or pt.y > self.h - 20 or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False

    def collision_with_self(self, pt=None):
        if pt is None:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, 20, 20))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, 20, 20))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    # move the head of the snake based on the action
    def _move(self, action):

        # get the current direction of the snake
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clock_wise.index(self.direction)

        # get the new direction based on the action
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[current_index] # straight
        elif np.array_equal(action, [0, 1, 0]):
            new_direction = clock_wise[(current_index + 1) % 4] # right turn
        elif np.array_equal(action, [0, 0, 1]):
            new_direction = clock_wise[(current_index - 1) % 4] # left turn

        self.direction = new_direction

        # get the new head position based on the direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 20
        elif self.direction == Direction.LEFT:
            x -= 20
        elif self.direction == Direction.DOWN:
            y += 20
        elif self.direction == Direction.UP:
            y -= 20
            
        self.head = Point(x, y)
            

