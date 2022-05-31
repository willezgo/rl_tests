print("Start RL Tests")

import gym
import time 

from stable_baselines3 import *
#from stable_baselines3.common.callbacks import *

ALGOS = [A2C, DDPG, DQN, PPO, SAC, TD3]

# Classic control
ENVS  = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v0']

# Box2D
ENVS = [
  #'LunarLander-v2', 
  #'LunarLanderContinuous-v2', 
  #'CarRacing-v0', # Needs a CNN
  'BipedalWalker-v3', 
  'BipedalWalkerHardcore-v3']


POLICIES = {
    'CarRacing-v0': "CnnPolicy"
}


def get_policy_for(env_id):
    if env_id in POLICIES:
        return POLICIES[env_id]

    # Default
    return "MlpPolicy"

def my_callback(lol, mdr):
    print("Callback:", type(lol), type(mdr))
    print("1:", lol)
    print("2:", mdr)
    exit()

def evaluate(env, model, ticks = 5000):
    obs = env.reset()
    total_reward = 0

    avg_sum   = 0
    avg_count = 0

    for i in range(ticks):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        #env.render("rgb_array")
        if done:
            obs = env.reset()
            avg_sum   += total_reward
            avg_count += 1
            total_reward = 0

    # Add the partial rewards of the last incomplete episode
    #avg_sum   += total_reward
    #avg_count += 1

    return (avg_sum / avg_count, avg_count)



def run_experiment(env_id, model_class, verbose = True):

    eval_env = gym.make(env_id)
    # eval_callback = EvalCallback(
    #     eval_env, 
    #     best_model_save_path='./logs/',
    #     log_path='./logs/', 
    #     eval_freq=500,
    #     deterministic=True, 
    #     render=False)

    env = gym.make(env_id)
    p = get_policy_for(env_id)

    model = model_class(p, env, verbose=0)

    if verbose:
        print("Start experiment with:", env_id, model_class)

    all_scores = []

    cycles = 40
    for i in range(cycles):
        model.learn(total_timesteps = 10000)
        (score, cnt) = evaluate(eval_env, model)
        all_scores.append(score)
        
        if verbose:
            pcnt = (i+1) / cycles * 100.0
            print(  "%.1f" % pcnt  , env_id, model_class, cnt, "\t", "%.2f" % score)
    
    env.close()
    eval_env.close()

    del model 
    del env 
    del eval_env

    return all_scores



def run_multialgo(env_id):
    print("Start multialgo experiment:", env_id)
    start = time.time()

    results = []

    for algo in ALGOS:
  
        try:
            scores = run_experiment(env_id, algo, verbose=True)
            results.append((max(scores), algo))

            print(env_id, algo, max(scores), scores)
        except Exception as e:
            print("EXCEPTION", e)
            pass
            #print(env_id, algo, type(e))

    results.sort(key= lambda x: x[0], reverse=True)

    end = time.time()
    duration = end - start 

    print("=== REPORT "+env_id+" in "+("%d" % duration)+" s ===")
    for (score, algo) in results:
        print("%.2f" % score, "\t", algo)
    
    if len(results) != 0:
        print("<BEST @ "+env_id+">:", results[0][1])
    else:
        print("<BEST @ "+env_id+">: n/a")


for env_id in ENVS:
    run_multialgo(env_id)
