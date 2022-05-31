import numpy as np
from gym import spaces

class EnvModule:
    """ Base class for environment modules. """
   
    def get_reward(self, env, reward):
        return reward

    def get_obs(self):
        """ Additional size to add to the observation vector. [(low, high)...] """
        return []

    def pre_step(self, env):
        pass 

    def post_step(self, env):
        pass 

    def pre_render(self, env):
        pass 

    def post_render(self, env):
        pass 

    def pre_reset(self, env):
        pass 

    def post_reset(self, env):
        pass 

    def observation(self, env, obs):
        """ Observation modifier hook, returns new observation. """
        #obs.append(0.42)
        return obs  


class EnvModuleCollection:
    """ Holds a collection of environment modules. """


    def __init__(self, env, mods = []):
        self._env = env
        self._mods = mods 
   

    def get_reward(self, env, reward):
        for i in self._mods:
            reward = i.get_reward(env, reward)
        return reward
    

    def get_observation_space(self, initial_low, initial_high):
        for i in self._mods:
            for (mod_low, mod_high) in i.get_obs():
                initial_low.append(mod_low)
                initial_high.append(mod_high)

        low  = np.array(initial_low ).astype(np.float32)
        high = np.array(initial_high).astype(np.float32)
        
        return spaces.Box(low, high)
  

    def call_observation(self, obs):
        for i in self._mods:
            obs = i.observation(self._env, obs)
        return obs 

    def call_pre_reset(self):
        for i in self._mods:
            i.pre_reset(self._env)

    def call_pre_step(self):
        for i in self._mods:
            i.pre_step(self._env)

    def call_pre_render(self):
        for i in self._mods:
            i.pre_render(self._env)

    def call_post_reset(self):
        for i in self._mods:
            i.post_reset(self._env)

    def call_post_step(self):
        for i in self._mods:
            i.post_step(self._env)

    def call_post_render(self):
        for i in self._mods:
            i.post_render(self._env)




