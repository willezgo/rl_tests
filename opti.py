# Optimize hyperparameters

from stable_baselines3.common.env_util import *
from stable_baselines3.common.evaluation import *
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.noise import *
from stable_baselines3 import *
import torch.nn.modules.activation
import random 
import numpy as np 
import gym
import time
import json

neurons = [32, 64, 128, 256, 512]

def choose_params():
    return {
        "learning_rate":   random.random() * 0.01,
        "buffer_size":     random.randint(100_000, 10_000_000),
        "learning_starts": random.randint(100, 10_000),
        "batch_size":      random.choice([32,64,128,256,512]),
        "gamma":           0.999 - (random.random() * 0.1),
        "tau":             random.random() * 0.01,
        "train_freq":      random.randint(1, 10),
        "action_noise_std":    0 if random.random() < 0.5 else random.random() * 0.2,
        "net_arch":            [random.choice(neurons) for _ in range(random.randint(1, 3))],
        "deterministic":       random.choice([False, True])

    }


### PARAMS
ENV_ID  = "BipedalWalkerHardcore-v3"
#SECONDS = 60 * 5
TICKS   = 100_000
P       = "MlpPolicy"
ALGOS   = [
   
]
###

print(f"Run experiments to optimize problem '{ENV_ID}' in {TICKS} ticks")

env_train = make_vec_env(ENV_ID)
env_eval  = make_vec_env(ENV_ID)

model = PPO("MlpPolicy", env_train,
    policy_kwargs = dict(net_arch=[256,256], activation_fn = torch.nn.modules.activation.LeakyReLU),
    
)
print(model.policy)

exit()





def learn_for(model, seconds):
    start = time.time()
    quanta = 25_000

    while time.time() - start < seconds:
        model.learn(quanta)
    
    
SCORES = []

while True:
    p = choose_params()
    print(p)

    for _ in range(1):
        model = TD3("MlpPolicy", env_train, verbose=0,
            learning_rate= p["learning_rate"],
            buffer_size  = p["buffer_size"],
            learning_starts= p["learning_starts"],
            batch_size = p["batch_size"],
            tau = p["tau"],
            gamma = p["gamma"],
            train_freq = p["train_freq"],
            action_noise = None if p["action_noise_std"] == 0 else NormalActionNoise(0, p["action_noise_std"]),
            policy_kwargs = dict(net_arch= p["net_arch"])
        )
        
        model.learn(TICKS)
        #learn_for(model, SECONDS)
        (score_mean, score_std) = evaluate_policy(model, env_eval, n_eval_episodes=20, deterministic = True) 
        print(score_mean, score_std)
        SCORES.append( (score_mean, score_std, p) )
        SCORES.sort(key = lambda x: x[0], reverse = True)

        with open("/out/opti.json", "w") as f:
            json.dump(SCORES, f, indent="\t")

    




