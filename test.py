from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy


env_id = 'PongNoFrameskip-v4'

print("Learn", env_id)




def create_env(env_id):
    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    env = make_atari_env(env_id, n_envs=4, seed=0)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)
    return env

env      = create_env(env_id)
eval_env = create_env(env_id)


model = PPO('CnnPolicy', env, verbose=0)
count_ticks = 0
batch_ticks = 1000
while True:
    model.learn(total_timesteps=batch_ticks)
    count_ticks += 1
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(count_ticks, "\t", "%.2f" % mean_reward)

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()






#import gym
#from stable_baselines3 import *
#import lib 

#Checkpoint every 100k / 1M
#Total: 50M ideally

#for env_id in lib.get_all_atari_ids():
#    env = lib.load_atari(env_id)
#    print(env_id, env, env.observation_space.shape)
#    model = PPO("CnnPolicy", env, verbose=1)
#    model.learn(total_timesteps = 1000*1000)


