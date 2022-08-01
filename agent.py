import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import torch
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:

    def __init__(self,w,h,newM):
        self.net = 0
        if self.net == 1:
            numFeat = int(11+w/20*h/20)
        else:
            numFeat = int(11)
        
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(numFeat, 256,64, 3,newM)
        self.trainer = QTrainer(self.model, gamma=self.gamma)
        self.n_games = self.model.n_games
        


            
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            ((dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d))),

            # Danger right
            ((dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d))),

            # Danger left
            ((dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d))),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            (game.food.x < game.head.x),  # food left
            (game.food.x > game.head.x),  # food right
            (game.food.y < game.head.y),  # food up
            (game.food.y > game.head.y)  # food down
            ]
        
        if self.net == 1:
            Matrix = game.getArray()
            return np.array(state+Matrix, dtype=int)
        else:
            return np.array(state, dtype=int)
        #Matrix = game.getArray()
        #return np.array(state+Matrix, dtype=int)
        #stateFull = state+Matrix
        #print(len(stateFull))


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        self.trainer.train_step(mini_sample)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_stepShort(state, action, reward, next_state, done)

        
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        #print(self.epsilon)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            #print("Random")
        else:
            state0 = state
            prediction = self.model.predOne(state0)
            move = np.argmax(prediction)
            final_move[move] = 1
            #print("Not Random")

        return final_move


def train(newM):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    w = 640
    h = 480
    agent = Agent(w,h,newM)
    game = SnakeGameAI(w,h)
    while True:
        # get old state

        state_old = agent.get_state(game)
        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model._save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train(True)