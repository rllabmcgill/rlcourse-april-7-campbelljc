import gym
import math
import numpy as np
from collections import defaultdict
from gym.envs.classic_control import CartPoleEnv

class Env():
    def __init__(self, name, num_actions, state_length, num_features=None):
        self.name = name
        self.num_actions = num_actions
        self.state_length = state_length
        self.num_features = state_length if num_features is None else num_features
    
    def new_episode(self):
        pass
    
    def end_episode(self):
        pass
    
    def should_end_episode(self):
        return False
    
    def reset(self):
        pass
    
    def save_stats(self):
        pass
    
    def step(self, action):
        raise NotImplementedError

class CartPole(Env):
    def __init__(self):
        self.gym = gym.make('CartPole-v0')
        
        # src: https://gym.openai.com/evaluations/eval_JeP6rWUQ8KuT8HB0YcR3g
        self.DIMS = 4
        self.N_TILES = 5
        self.N_TILINGS = 16
        self.TILES_START = np.array([-2.5, -4, -0.25, -3.75], dtype=np.float64)
        self.TILES_END   = np.array([ 2.5,  4,  0.25,  3.75], dtype=np.float64)
        self.TILES_RANGE = self.TILES_END - self.TILES_START
        self.TILES_STEP = (self.TILES_RANGE / (self.N_TILES * self.N_TILINGS))
        
        super().__init__('gym_cartpole', num_actions=2, num_features=self.N_TILINGS * (self.N_TILES ** self.DIMS) * 2, state_length=-1)
    
    def get_features_for_state_action(self, state, action=None):
        # src: https://gym.openai.com/evaluations/eval_JeP6rWUQ8KuT8HB0YcR3g
        indices = [np.floor(self.N_TILES*(state - self.TILES_START + (i * self.TILES_STEP))/self.TILES_RANGE).astype(np.int) for i in range(self.N_TILINGS)]

        flattened_indices = np.array([np.ravel_multi_index(index, dims=tuple([self.N_TILES] * self.DIMS), mode='clip') for index in indices])
        if action is not None:
            flattened_indices += int(action * (self.N_TILINGS * (self.N_TILES ** self.DIMS)))
        
        return flattened_indices
    
    def new_episode(self):
        return self.gym.reset()
    
    def step(self, action):
        state, reward, done, _, _ = self.gym.step(action)
        return state, reward, done
