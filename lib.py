import time
import gym
import gym.wrappers
import atari_py as ap

from stable_baselines3.common.vec_env import *
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper

def set_title(title):
    print(f'\33]0;{title}\a', end='', flush=True)



def record_video(model, env, deterministic, video_length, video_folder, name_prefix):
  obs = env.reset()

  # Record the video starting at the first step
  env = VecVideoRecorder(env, 
                        video_folder,
                        record_video_trigger=lambda x: x == 0, 
                        video_length=video_length,
                        name_prefix=name_prefix)

  obs = env.reset()
  for _ in range(video_length + 1):
    (action, _) = model.predict(obs, deterministic = deterministic)
    #action = [env.action_space.sample()]
    obs, reward, _, _ = env.step(action)
    #print("Reward:", reward)
  # Save the video
  env.close()

class Throttle:
    def __init__(self, seconds):
        self.seconds = seconds 
        self.next    = 0

    def tick(self):
        now = time.time()
        trigger = now >= self.next 

        if trigger:
            self.next = now + self.seconds

        return trigger 




class VecRewardOffset(VecEnvWrapper):
    """
    Adds an offset to the reward at each time step.
    For example, this can be used to deduct a penalty
    to incentivize agents to move forward quickly.
    """

    def __init__(self, venv, reward_offset):
        super().__init__(venv)
        print("Initialize VecRewardOffset:", reward_offset)
        self.reward_offset = reward_offset


    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        reward += self.reward_offset
        return obs, reward, done, info


def get_all_atari_ids(no_frameskip = False):
    o = []

    return ["Pong-v4"]

    all_envs = gym.envs.registry.all()
    all_envs = [i.id for i in all_envs]

    all_envs = set(all_envs)

    for i in ap.list_games():
        chunks = i.split("_")
        chunks = [i.capitalize() for i in chunks]
        j = ''.join(chunks) + ("NoFrameskip" if no_frameskip else "") + "-v4"
       
        # Doesn't work
        if j == "Defender-v4":
            continue

        if j not in all_envs:
            continue

        o.append(j)
        

    o.sort()
    return o 


def load_atari(env_id):
    e = gym.make(env_id)
    # TODO: re-enable and disable grayscale 
    #e = gym.wrappers.AtariPreprocessing(e, frame_skip=1)
    return e


