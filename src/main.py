from env import *
from agents import *
from policy import *
#from gym.envs.classic_control import MountainCarEnv

import itertools
import numpy as np
from keras import initializers

if __name__ == '__main__':    
    #env = GymWrapper(MountainCarEnv(), "gym_mountaincar", num_features=2)
    env = CartPole()
    
    num_episodes = 2000
    num_steps_per_episode = 500
    
    num_evaluations = 50
    num_episodes_per_evaluation = 5
    
    param_combos = list(itertools.product(
        [FAQAgent], #, OneStepActorCriticAgent], # agents
        [0.995], # gamma
        [0.01*i for i in range(1, 10)], # alpha
        [0.0005 + 0.0001*i for i in range(1, 5)], # beta
        [0.05*i for i in range(0, 5)] # lambda
    ))
    
    for j, combo in enumerate(param_combos):
        agent_fn, gamma, alpha, beta, lambda_ = combo
        print(gamma, alpha, beta, lambda_)
        agent = agent_fn(env=env, policy=GreedyQPolicy(), gamma=gamma, alpha=alpha, beta=beta, lambda_=lambda_)
        eval_agent = agent_fn(env=env, policy=GreedyQPolicy(), gamma=gamma, alpha=alpha, beta=beta, lambda_=lambda_)
        eval_agent.set_testing()
        
        if os.path.exists(agent.savedir + '/records.dll'): continue
        
        for ep in range(num_episodes+1):
            print("Combo", j, "episode", ep)
            agent.new_episode()
            state = env.new_episode()
            agent.prev_state = state
            while agent.cur_action_this_episode < num_steps_per_episode:
                try:
                    action = agent.choose_action(state)
                    state, reward, terminal = env.step(action)
                    agent.update(state, reward, terminal)
                    agent.end_turn(state)
                except RangeError:
                    input("range")
                    break
                if terminal:
                    break
            agent.end_episode()
            env.end_episode()
        
            if ep % (num_episodes / num_evaluations) == 0:
                # time to evaluate target policy.
                for eval_ep in range(num_episodes_per_evaluation):
                    # copy the q-values. (start each ep fresh)
                    eval_agent.set_params(*agent.get_params())
            
                    print("Agent", j, ", episode", eval_ep, "(evaluation)")
                    eval_agent.new_episode()
                    state = env.new_episode()
                    eval_agent.prev_state = state
                    while eval_agent.cur_action_this_episode < num_steps_per_episode:
                        try:
                            action = eval_agent.choose_action(state)
                            state, reward, terminal = env.step(action)
                            eval_agent.update(state, reward, terminal, update=False)
                            eval_agent.end_turn(state)
                        except RangeError:
                            input("range")
                            break
                        if terminal:
                            break
                    eval_agent.end_episode()
                    env.end_episode()
                eval_agent.avg_last_episodes(ep, num_episodes_per_evaluation)
        eval_agent.save_stats()