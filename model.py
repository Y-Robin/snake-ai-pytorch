import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from tensorflow import keras
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import convert_to_tensor, shape
from tensorflow.keras.models import load_model
import time



class Linear_QNet:
    def __init__(self, input_size, hidden_size,hidden_size2, output_size,newM):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.001
        if newM:
            self.model = load_model("preTrained.h5")
            print("old_model")
            self.n_games = 100
        else:
            self.model = self._build_model(hidden_size,hidden_size2)
            print("new_model")
            self.n_games = 0
        
    def _build_model(self,hidden_size,hidden_size2):
        
        model = Sequential() 
        model.add(keras.Input(shape=(self.input_size,)))
        model.add(Dense(hidden_size, activation="relu"))
        model.add(Dense(hidden_size2, activation="relu"))
        model.add(Dense(self.output_size, activation="linear"))
        model.compile(loss="mse",
                     optimizer=Adam(learning_rate=self.learning_rate))
        
        model.summary()
        return model
        

    def predOne(self,state):

        state = convert_to_tensor(np.array(state,ndmin=2))
        pred = self.model(state)
        return pred.numpy()
    
    def _save(self):
        self.model.save("preTrained.h5")

        

class QTrainer:
    def __init__(self, model, gamma):
        self.gamma = gamma
        self.model = model
        self.n_games = self.model.n_games




    def train_step(self,minibatch):
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        #print(len(dones))
        for i in range(len(dones)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            target = reward # if done 
            
            if not done:
                target = (reward +
                        self.gamma *
                        np.amax(self.model.predOne(next_state)[0]))

            target_f = self.model.predOne(state)
            #print(time.time() - t)
            target_f[0][action] = target
            #print(target_f)
            if i == 0:
                state = np.array(state,ndmin=2)
                stateAll = state
                targetAll = target_f
            else:
                state = np.array(state,ndmin=2)
                stateAll = np.concatenate((stateAll,state),axis=0)
                targetAll = np.concatenate((targetAll,target_f),axis=0)

        self.model.model(stateAll, targetAll) 

            
    def train_stepShort(self,state, action, reward, next_state, done):
        target = reward # if done 
        if not done:
            target = (reward +
                    self.gamma *
                    np.amax(self.model.predOne(next_state)[0]))

        target_f = self.model.predOne(state)
        #print(target_f)
        target_f[0][np.argmax(action)] = target
        #print(target_f)
        state = np.array(state,ndmin=2)

        self.model.model.fit(state, target_f, epochs=1, verbose=0) 
    



