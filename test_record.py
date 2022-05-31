import os, sys, time
import gym
import highway_env
from utils import *
import torch.nn.modules.activation
import stable_baselines3.common.monitor
from stable_baselines3.common.env_util import *
from stable_baselines3.common.evaluation import *
from stable_baselines3.common.vec_env import *
from stable_baselines3.common.noise import *
from stable_baselines3 import *
import lib
import mybipedal 

# Define custom environment strings
custom_envs = {
  "MyBipedalWalkerHardcore-penalty50": lambda: mybipedal.make_mybipedal(hardcore=True, reward_failure=-50),
  "MyBipedalWalkerHardcore-penalty25": lambda: mybipedal.make_mybipedal(hardcore=True, reward_failure=-25),
  "MyBipedalWalkerHardcore-penalty12": lambda: mybipedal.make_mybipedal(hardcore=True, reward_failure=-12),
  "MyBipedalWalkerHardcore-penalty0" : lambda: mybipedal.make_mybipedal(hardcore=True, reward_failure=  0),
  
  "MyBipedP25Doll": lambda: mybipedal.make_mybipedal(hardcore=False, reward_failure=-25, doll_mod = True),
  "MyBipedHardP25Doll": lambda: mybipedal.make_mybipedal(hardcore=True, reward_failure=-25, doll_mod = True),
}


# Set of all environments associated to an index
e_set = [
  'LunarLanderContinuous-v2', # 0
  'BipedalWalker-v3',         # 1
  'BipedalWalkerHardcore-v3', # 2
  'MyBipedalWalkerHardcore-penalty50', # 3
  'MyBipedalWalkerHardcore-penalty25', # 4
  'MyBipedalWalkerHardcore-penalty12', # 5
  'MyBipedalWalkerHardcore-penalty0' , # 6

  "MyBipedP25Doll",       # 7
  "MyBipedHardP25Doll",   # 8

  'parking-v0',               
  'racetrack-v0'              
]

# FPS of particular environments
e_fps = {
  'parking-v0': 10,
  'racetrack-v0': 10
}

algo_set = {
  "td3": TD3,
  "sac": SAC,
  "ddpg": DDPG,
  ###
  "ppo": PPO,
  "a2c": A2C
}

IDX = int(sys.argv[1])

IS_ATARI = False # TODO: FIXME

n_envs = 1
env_id = e_set[IDX]
video_folder = '/out/exp_doll8_stoch/'
video_length = e_fps.get(env_id, 50) *60*2

DO_TRAIN   = True
DO_BREAK   = False

IS_DETERMINISTIC = False
ALGO_ID = "sac"
ALGO   = algo_set[ALGO_ID]
POLICY = 'MlpPolicy' if not IS_ATARI else 'CnnPolicy'
TICKS      =  50_000
SAVE_EVERY = 250_000

POLICY_KWARGS = dict(
  net_arch = [256, 256], #400,300
  #activation_fn = torch.nn.modules.activation.ReLU
)

if ALGO_ID == "ppo":
  n_envs = 16
  TICKS      =   500_000
  SAVE_EVERY = 2_000_000 

if ALGO_ID == "a2c":
  n_envs = 32
  TICKS      =   500_000
  SAVE_EVERY = 3_000_000

if env_id == "parking-v0":
  POLICY = 'MultiInputPolicy'

# On BipedalHard + LunarContinuous:
# 100k = 5 minutes


RUN_ID = str(int(time.time() * 1000))


print("Training & recording:", env_id)
print("Run id:", RUN_ID)
print("n_envs:", n_envs)

os.nice(19)
lib.set_title(f"Training: {video_folder} / {ALGO_ID} / {env_id} / {RUN_ID}")

def make_racetrack():
  print("Make racetrack")
  e = gym.make("racetrack-v0")
  e.configure({
    "observation": {
        "type": "Kinematics"
    }
  })
  e.reset()
  return e

def init_env(env_id, n_envs):
  if env_id == "racetrack-v0":
    env =  DummyVecEnv([lambda: make_racetrack()])
    return env
  
  # First, try to see if it corresponds to a custom environment
  if env_id in custom_envs:
    env = DummyVecEnv([  
        lambda: stable_baselines3.common.monitor.Monitor(custom_envs[env_id]()) 
    ] * n_envs)
  
  elif IS_ATARI:
    env = make_atari_env(env_id, n_envs=n_envs)
    env = VecFrameStack(env, n_stack=4)
  else:
    #env = DummyVecEnv([lambda: gym.make(env_id)] * n_envs)
    env = make_vec_env(env_id, n_envs)
  
  #env = VecNormalize(env)
  #env = lib.VecRewardOffset(env, -0.5)

  return env 



# Separate env for evaluation/recording
train_env = init_env(env_id, n_envs)
test_env  = init_env(env_id, 1)

print("Obs:", test_env.observation_space)
print("Act:", test_env.action_space)

print("Train env:", train_env.num_envs, train_env)
print("Test  env:", test_env.num_envs, test_env)

#model = ALGO(POLICY, train_env,
#  gamma   = 0.995, 
#  verbose = 0)

if ALGO_ID == "td3" or ALGO_ID == "sac":
  # Hyperparams for BipedalWalkerHardcore (TD3/MlpPolicy)
  model = ALGO(POLICY, train_env, verbose=1,
    gamma= 0.99,
    buffer_size= 1000000,
    learning_starts= 10000,
    #noise_type= 'normal'
    #noise_std= 0.1
    action_noise = NormalActionNoise(0, 0.1),
    batch_size= 256,
    train_freq= 1,
    learning_rate= 7e-4, #"lin_7e-4",
    policy_kwargs= POLICY_KWARGS) 
elif ALGO_ID == "ppo":
  model = ALGO(POLICY, train_env, verbose=1,
    n_steps= 2048,
    batch_size= 64,
    gae_lambda= 0.95,
    gamma= 0.99,
    n_epochs= 10,
    ent_coef= 0.001,
    learning_rate= 2.5e-4,
    clip_range= 0.2,
    policy_kwargs= POLICY_KWARGS)
elif ALGO_ID == "a2c":
  model = ALGO(POLICY, train_env, verbose=1,
    ent_coef= 0.001,
    max_grad_norm= 0.5,
    n_steps= 8,
    gae_lambda= 0.9,
    vf_coef= 0.4,
    gamma= 0.99,
    use_rms_prop= True,
    normalize_advantage= False,
    learning_rate= 0.0008,
    use_sde= True,
    policy_kwargs= dict(log_std_init=-2, ortho_init=False)
  )
else:
  raise Exception()
  model = TD3.load("models/model_MyBipedalWalkerHardcore-penalty0_1642706161946.pkl", env = train_env)

print("Model:", model)
print("Policy kwargs:", model.policy_kwargs)
print("Policy:", model.policy)
#for param in model.policy.mlp_extractor.shared_net.parameters():
#  print("\t", type(param), param.size())


os.makedirs(video_folder, exist_ok =  True)
f_csv = open(video_folder + f"report_{ALGO_ID}_{IS_DETERMINISTIC}_" + env_id + "_" + RUN_ID + ".csv", "w")

f_csv.write('\t'.join(["Env", "Algo", "Determ.", "Ticks", "Time", "Mean", "U", "D", "Std"])  + "\n")

start_time = time.time()
it = 0
while True:
  (score, score_std) = evaluate_policy(model, test_env, n_eval_episodes = 20, deterministic = IS_DETERMINISTIC)
  
  if it % SAVE_EVERY == 0:
    model.save(video_folder + f"model_{ALGO_ID}_" + env_id + "_" + RUN_ID + ".pkl")
    print("Start recording video...")
    lib.record_video(model, test_env, IS_DETERMINISTIC, video_length, video_folder, f"vid_{ALGO_ID}_{env_id}-{RUN_ID}-{it}")

  wall_clock = (time.time() - start_time) / (60 * 60 * 24)
  chunks = [env_id, ALGO_ID, IS_DETERMINISTIC, it, wall_clock, score, score + score_std, score - score_std, score_std]

  line = '\t'.join([str(i) for i in chunks]) 
  print(line)
  f_csv.write(line + "\n")
  f_csv.flush()

  if DO_TRAIN:
    model.learn(total_timesteps=TICKS)
    it += TICKS
  if DO_BREAK:
    break
