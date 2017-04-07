import os
import dill
from env import *
from fileio import *
import seaborn as sns
import matplotlib.pyplot as plt 
from collections import namedtuple
from gym.envs.classic_control import MountainCarEnv

Episode = namedtuple('Episode', 'game_id num_actions total_reward')

if __name__ == '__main__':
    if not os.path.exists('../figs'):
        os.makedirs('../figs')
        
    envs = [CartPole()]    
    for env in envs:
        results = []
        for subdir in get_immediate_subdirectories('../results'):
            if env.name not in subdir:
                continue
            model_name = subdir[len(env.name)+7:]
        
            # load the results for this model
            if not os.path.exists("../results/"+subdir+"/records.dll"): continue
            records = dill.load(open("../results/"+subdir+"/records.dll", 'rb'))
        
            # determine average cumulative reward for this model
            avg_total_reward = sum([rec.total_reward for rec in records])/len(records)
            results.append((model_name, avg_total_reward))
        
        results.sort(key=lambda tup: tup[1])
        for i, j in results:
            print(i, j)
        
        # graph the avg total reward
        if not os.path.exists('../figs'):
            os.makedirs('../figs')
    
        plt.figure()
        plt.bar(np.arange(len(results)), [r[1] for r in results], align='center')
        plt.xticks(np.arange(len(results)), [r[0] for r in results])
        plt.ylabel('Average total reward')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_avgtotalreward.png')
        
        best_results = defaultdict(lambda: [])
        results = []
        plt.figure()
        num_recs = 0
        for subdir in get_immediate_subdirectories('../results'):
            if env.name not in subdir:
                continue
            model_name = subdir[len(env.name)+7:]
    
            # load the results for this model
            if not os.path.exists("../results/"+subdir+"/records.dll"): continue
            records = dill.load(open("../results/"+subdir+"/records.dll", 'rb'))
    
            # determine average cumulative reward for this model
            avg_total_reward = sum([rec.total_reward for rec in records])/len(records)
            plt.plot([rec.game_id for rec in records], [rec.total_reward for rec in records], label=model_name)
            
            model_name_stripped = model_name.split("_")[0]
            if avg_total_reward > 75 and np.std(np.array([rec.total_reward for rec in records])) < 50:
                #best_results[model_name_stripped] = (avg_total_reward, records, model_name, model_name_stripped)
                best_results[model_name_stripped].append((records, model_name))
                num_recs += 1
        
        plt.legend()
        plt.ylabel('Total reward over time')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_totalreward.png')
        
        sns.set_palette(sns.color_palette("hls", num_recs))
        plt.figure()
        axes = plt.gca()
        axes.set_ylim([0, 300])
        axes.set_xlim([0, 2000])
                    
        for model_name_stripped in best_results:
            for records, model_name in best_results[model_name_stripped]:
                plt.plot([rec.game_id for rec in records], [rec.total_reward for rec in records], label=model_name)
        
        plt.legend()
        plt.ylabel('Total reward over time')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_totalreward_bestparams.png')
        
        '''
        best_params = [
            
        ]
        
        plt.figure()
        
        for model_name_stripped in best_results:
            for records, model_name in best_results[model_name_stripped]:
                if model_name in best_params:
                    plt.plot([rec.game_id for rec in records], [rec.total_reward for rec in records], label=model_name)
        
        plt.legend()
        plt.ylabel('Total reward over time')
        plt.title(env.name)
        plt.savefig('../figs/' + env.name + '_totalreward_bestparams_final.png')
        '''