import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

from model import Linear_Qnet, QTrainer

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 10
Learning_Rate = 0.001

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.gamma   = 0.8 
        self.model   = Linear_Qnet(16, 512, 3)
        self.trainer = QTrainer(self.model, lr=Learning_Rate, gamma=self.gamma)
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-20, self.head.y),
                      Point(self.head.x-(2*20), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def get_state(self):
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
        
        
    def play_step(self):
        # 1. collect user input
        old_direction = self.direction
        action = [0,0,0]
        reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        if old_direction == Direction.LEFT:
            if self.direction == Direction.LEFT:
                action[0] = 1 # keep going forward
            elif self.direction == Direction.UP:
                action[1] = 1 # turn right
            elif self.direction == Direction.DOWN:
                action[2] = 1 # turn left

        elif old_direction == Direction.RIGHT:
            if self.direction == Direction.RIGHT:
                action[0] = 1 # keep going forward
            elif self.direction == Direction.UP:
                action[2] = 1
            elif self.direction == Direction.DOWN:
                action[1] = 1

        elif old_direction == Direction.UP:
            if self.direction == Direction.UP:
                action[0] = 1
            elif self.direction == Direction.LEFT:
                action[2] = 1
            elif self.direction == Direction.RIGHT:
                action[1] = 1

        elif old_direction == Direction.DOWN:
            if self.direction == Direction.DOWN:
                action[0] = 1
            elif self.direction == Direction.LEFT:
                action[1] = 1
            elif self.direction == Direction.RIGHT:
                action[2] = 1     


        
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            reward = -10
            return game_over, self.score, action, reward
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score, action, reward
    
    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def train_short_memory(self, state, action, reward, next_state,  done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

    
            

if __name__ == '__main__':
    game = SnakeGame()
    
    high_score = 0

    # game loop
    while True:

        start_state = game.get_state()
        game_over, score, action, reward = game.play_step()
        end_state = game.get_state()

        # train short term memory
        game.train_short_memory(start_state, action, reward, end_state, game_over)
        
        if game_over == True:
            game.reset()

            if score > high_score:
                high_score = score
                print("New high score: ", high_score)

                game.model.save()
        
    print('Final Score', score)
        
        
    pygame.quit()