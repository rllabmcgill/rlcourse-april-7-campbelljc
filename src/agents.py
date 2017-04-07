import os, math
import numpy as np
import dill, pickle
from graphs import Episode
from functools import partial
from collections import defaultdict
from fileio import append, write_list, read_list

class Agent():
    def __init__(self):
        self.training = True
        self.cur_episode = 0
        self.cur_action = 0
        self.game_records = []        
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
    
    def set_testing(self):
        self.training = False
    
    def save_stats(self):
        write_list([self.cur_episode, self.cur_action], self.savedir+"stats")
        with open(self.savedir+"records.dll", 'wb') as output:
            dill.dump(self.game_records, output)
    
    def new_episode(self):
        self.total_reward = 0
        self.cur_action_this_episode = 0
    
    def end_turn(self):
        self.cur_action += 1
        self.cur_action_this_episode += 1
    
    def end_episode(self):
        if not self.training:
            self.game_records.append(Episode(self.cur_episode, self.cur_action_this_episode, self.total_reward))
        print(self.cur_episode, self.cur_action_this_episode, self.total_reward)
        self.cur_episode += 1
    
    def avg_last_episodes(self, eval_id, num_to_avg):
        records = self.game_records[-num_to_avg:]
        self.game_records = self.game_records[:-num_to_avg]
        
        avg_actions = sum([rec.num_actions for rec in records])/len(records)
        avg_total_reward = sum([rec.total_reward for rec in records])/len(records)
        self.game_records.append(Episode(eval_id, avg_actions, avg_total_reward))

class QAgent(Agent):
    def __init__(self, env, policy, alpha, gamma, name):
        self.policy = policy
        if policy is not None:
            self.policy.agent = self
        self.alpha = alpha
        self.gamma = gamma
        self.num_actions = env.num_actions
        self.state_length = env.state_length
        self.env = env
        
        self.savedir = '../results/model_' + env.name + '_' + name + '_' + self.get_param_str() + '/'
        super().__init__()
    
    def get_param_str(self):
        return 'g' + str(self.gamma) + '_a' + str(self.alpha)
    
    def save_stats(self):
        super().save_stats()    
        write_list([self.alpha, self.gamma], self.savedir+"params")
    
    def new_episode(self):
        super().new_episode()
        self.state = None
        self.prev_state = None
        self.last_action = None
        
    def end_turn(self, state):
        self.prev_state = np.copy(state)
        super().end_turn()
        
    def choose_action(self, state):
        action = self.policy.select_action(q_values=self.get_qvals_for_state(state))
        self.last_action = action
        return action
        
    def update(self, state, reward, terminal):
        raise NotImplementedError

class TabularQAgent(QAgent):
    def __init__(self, env, policy, alpha, gamma, name='QLearningTabular'):
        self.qvals = defaultdict(lambda: [0] * env.num_actions)
        if 'get_initial_qvalues' in dir(env):
            self.qvals = env.get_initial_qvalues()
        
        super().__init__(env, policy, alpha, gamma, name)
    
    def get_qvals_for_state(self, state):
        return self.qvals[tuple(state)]
    
    def predict(self, state):
        return max(self.qvals[tuple(state)])
    
    def update(self, state, reward, terminal):
        if self.cur_action_this_episode == 0:
            return # skip initial state (haven't chosen an action yet)
        
        assert self.last_action is not None
        assert self.prev_state is not None
        
        old_state = self.prev_state
        action = self.last_action
        new_state = state
        
        self.total_reward += reward
        
        if terminal:
            update = reward # tt = rr
        else: # non-terminal state
            newQ = self.qvals[tuple(new_state)][:]
            maxQ = np.max(newQ) # maxQ = max_a' Q(ss', a') for all a' in A
            update = (reward + (self.gamma * maxQ)) # tt = rr + gamma * max_a' Q(ss', a') for all a' in A
        
        cur_qval = self.qvals[tuple(old_state)][action]
        alpha = 1/((self.state_counts[(tuple(old_state), action)] + 1) ** self.alpha)
        self.qvals[tuple(old_state)][action] = ((1-alpha)*cur_qval) + (alpha*update)
                
        self.state_counts[(tuple(old_state), action)] += 1

class FAQAgent(QAgent):
    def __init__(self, env, policy, alpha, gamma, name='FAQAgent', **args):
        super().__init__(env, policy, alpha, gamma, name)
        self.num_features = env.num_features
        self.weights = np.random.normal(0, 0.1, self.num_features)
    
    def get_params(self):
        return [np.copy(self.weights)]
    
    def set_params(self, weights):
        self.weights = weights
    
    def get_qvals_for_state(self, state):
        return [self.get_qvalue(state, i) for i in range(self.num_actions)]
    
    def predict(self, state):
        return np.argmax([self.get_qvalue(state, act) for act in range(self.num_actions)])
    
    def get_qvalue(self, state, action):
        #state_features = self.env.get_features_for_state(state) #np.transpose(np.dot(np.transpose(self.features), state))
        #return np.dot(self.weights[action], state_features)
        return np.sum(self.weights[self.env.get_features_for_state_action(state, action)])
    
    def update(self, next_state, reward, terminal, update=True):
        assert self.last_action is not None
        assert self.prev_state is not None
        
        self.total_reward += reward
        
        if not update:
            return
        
        action = self.last_action
        
        predicted_qvalue = self.get_qvalue(self.prev_state, action)
        
        if terminal:
            td_error = reward - predicted_qvalue
        else:
            best_next_action = self.predict(next_state)
            predicted_next_qval = self.get_qvalue(next_state, best_next_action)
            td_error = reward + (self.gamma * predicted_next_qval) - predicted_qvalue
        
        state_features = self.env.get_features_for_state_action(self.prev_state, action)
        ew = np.zeros((self.num_features,))
        for feat in state_features:
            ew[feat] += 1
        
        self.weights += (self.alpha * td_error) * ew

class SarsaLambdaAgent(QAgent):
    def __init__(self, env, policy, alpha, gamma, lambda_, name='SarsaLambdaAgent', **args):
        self.lambda_ = lambda_
        super().__init__(env, policy, alpha, gamma, name)
        self.num_features = env.num_features
        
        self.V = np.random.normal(0, 0.1, self.num_features) # np.zeros((self.num_features))
        self.e = np.zeros((self.num_features))
        self.next_action = None
    
    def get_param_str(self):
        return super().get_param_str() + '_l' + str(self.lambda_)
    
    def get_params(self):
        return np.copy(self.V), np.copy(self.e)
    
    def set_params(self, v, e):
        self.V = v
        self.e = e
    
    def new_episode(self):
        super().new_episode()
        print("new ep, setting NA")
        self.next_action = np.random.choice([i for i in range(self.num_actions)])
    
    def get_qvals_for_state(self, state):
        return [self.get_qvalue(state, i) for i in range(self.num_actions)]
    
    def predict(self, state):
        return np.argmax([self.get_qvalue(state, act) for act in range(self.num_actions)])
    
    def choose_action(self, state):
        assert self.next_action is not None
        print("c", self.next_action)
        self.last_action = self.next_action
        return self.last_action
    
    def get_qvalue(self, state, action):
        return np.sum(self.V[self.env.get_features_for_state_action(state, action)])
        
    def grad(self, state, action):
        return self.env.get_features_for_state_action(state, action)
    
    def update(self, next_state, reward, terminal, update=True):
        assert self.last_action is not None
        assert self.prev_state is not None
        
        self.next_action = None
        self.total_reward += reward
        
        if not update:
            input("..")
            print(self.next_action)
            return
        
        action = self.last_action
        
        predicted_qvalue = self.get_qvalue(self.prev_state, action)
        
        if terminal:
            td_error = reward - predicted_qvalue
            self.next_action = None
        else:
            best_next_action = self.predict(next_state)
            self.next_action = best_next_action
            predicted_next_qval = self.get_qvalue(next_state, best_next_action)
            td_error = reward + (self.gamma * predicted_next_qval) - predicted_qvalue
                
        prev_state_features = self.env.get_features_for_state_action(self.prev_state, self.last_action)
        for feat in prev_state_features:
            self.e[feat] += 1

        self.V += self.alpha * td_error * self.e
        self.e *= (self.gamma * self.lambda_)

# src: https://github.com/gauthamvasan/OpenAI-Gym/blob/master/Cartpole/linear_ACRL.py
def gibbs_action_sampler(features, num_actions, u):
    action_prob = np.zeros(num_actions)  # Action probabilities
    gibbs_den = 0     # gibbs policy denominator
    gibbs_num = []
    for i in range(num_actions):
        val = 0
        for f in features:
            val += u[f,i]
        gibbs_num.append(val)
        gibbs_den += math.exp(val)

    for i in range(num_actions):
        prob = (math.exp(gibbs_num[i]))/gibbs_den
        action_prob[i] = prob

    return action_prob

# ref: https://github.com/gauthamvasan/OpenAI-Gym/blob/master/Cartpole/linear_ACRL.py
class OneStepActorCriticAgent(QAgent):
    def __init__(self, env, policy, alpha, beta, gamma, lambda_, name='1StepAC'):
        self.beta = beta
        self.lambda_ = lambda_       
        super().__init__(env, policy, alpha, gamma, name)

        self.num_features = env.num_features

        self.w = np.zeros((self.num_features,self.num_actions))
        self.u = np.random.normal(0, 0.1, (self.num_features,self.num_actions))
        self.ew = np.zeros((self.num_features,self.num_actions))
        self.eu = np.random.normal(0, 0.1, (self.num_features,self.num_actions))
    
    def get_param_str(self):
        return super().get_param_str() + '_b' + str(self.beta) + '_l' + str(self.lambda_)
    
    def get_params(self):
        return np.copy(self.w), np.copy(self.u), np.copy(self.ew), np.copy(self.eu)
    
    def set_params(self, w, u, ew, eu):
        self.w = w
        self.u = u
        self.ew = ew
        self.eu = eu
    
    def new_episode(self):
        super().new_episode()
        self.next_action = None
    
    def value(self, features, action):
        Val = 0.0
        for index in features:
            Val += self.w[index,action]
        return Val
    
    def choose_action(self, state, t = 1.0):
        if self.next_action is not None:
            self.last_action = self.next_action
            self.next_action = None
            return self.last_action            
            
        state_features = self.env.get_features_for_state_action(self.prev_state)
        self.action_prob = gibbs_action_sampler(state_features, self.num_actions, self.u)
        
        self.last_action = np.where(self.action_prob.cumsum() >= np.random.random())[0][0]
        return self.last_action

    def sample_action(self, action_prob):
        return np.where(action_prob.cumsum() >= np.random.random())[0][0]
    
    def update(self, next_state, reward, terminal, update=True):
        assert self.last_action is not None
        assert self.prev_state is not None
        
        self.total_reward += reward
        
        if not update:
            return
        
        prev_state_features = self.env.get_features_for_state_action(self.prev_state, self.last_action)
        next_state_features = self.env.get_features_for_state_action(next_state, self.choose_action(next_state))
        self.action_prob = gibbs_action_sampler(prev_state_features, self.num_actions, self.u)
        next_action_prob = gibbs_action_sampler(next_state_features, self.num_actions, self.u)
        
        cur_state_val = self.value(prev_state_features, self.last_action)
        delta = reward - cur_state_val
        
        next_action = self.sample_action(next_action_prob)
        self.next_action = next_action
        next_state_val = self.value(next_state_features, next_action)
        delta += self.gamma * next_state_val
        
        self.ew = self.gamma * self.lambda_ * self.ew
        for index in prev_state_features:
            self.ew[index, self.last_action] += 1
        
        self.w += self.alpha * delta * self.ew
        
        compatFeats = np.zeros((self.num_features,self.num_actions))
        for f in prev_state_features:
            compatFeats[f,self.last_action] = 1 - self.action_prob[self.last_action]

        for i in range(self.num_actions):
            sample_features_bits = np.zeros((self.num_features, self.num_actions))
            if i != self.last_action:
                for f in prev_state_features:
                    sample_features_bits[f,i] = self.action_prob[i]
                compatFeats -= sample_features_bits
        
        self.eu = self.gamma * self.lambda_ * self.eu
        self.eu += compatFeats
        
        self.u += self.beta * delta * self.eu
