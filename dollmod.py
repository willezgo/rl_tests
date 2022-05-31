# Implement a "Red-Light Green-Light" game

# DONE: Basic game cycle (move, can't move)
# DONE: Movement detection
# DONE: Shooting
# DONE: Add the game cycle to the observation vector

# What happened during the frame?
# If green light:
    # Made progress (new chunked X value): +1
    # Didn't make progress: 0
    # Regressed: 0
    # Fell: -25
# If red light:
    # Not shot: +1
    # Got shot: -25

import random 
from envmod import EnvModule

FPS = 50

DOLL_MOVE_MIN = 1 * FPS
DOLL_MOVE_MAX = 10 * FPS

DOLL_STILL_MIN = 1 * FPS
DOLL_STILL_MAX = 2 * FPS

DOLL_SHOOT_FORCE = (-10_000, 0)

COLOR_KICKED = (1.0, 0.5, 0.0)

class DollMod(EnvModule):
    def __init__(self):
        self.doll_can_move = True
        self.doll_tick_remaining = 0
        self.doll_tick_duration  = 0

        self.immunity = 0

        # Last X position of the hull
        self.last_x = 0

        # Movements of the hull and all 4 legs
        self.mov_norm = [0]    * 5   # normalized 

        self.log = []
      
    
    def get_reward(self, env, reward):
        cur_x = int(env.hull.position.x) 
        reward = 0
        
        delta = int(cur_x) - int(self.last_x)

        if self.doll_can_move:
            if delta > 0:
                reward += delta 
                self.last_x = cur_x
        
        # Failure penalty
        if env.game_over or self.motion_detected:
            reward += env.reward_failure
       

        #else:
        #    if self.immunity <= 0:
        #        reward += .1

        #print("R", self.doll_can_move, reward)
        
        return reward

    def get_obs(self):
        """ Additional size to add to the observation vector. [(low, high)...] """
        return [(0, 1), (0, 1)]

    def observation(self, env, obs):
        obs.append(self._compute_obs())
        obs.append(self._compute_obs_speed())
        return obs

    ### movement detection

    def _do_movement_detection(self, env):
        self._check_movement(env.hull, 0)

        for leg_id in range(len(env.legs)):
            self._check_movement(env.legs[leg_id], 1 + leg_id)

   


    def _check_movement(self, obj, idx):
        m = self._movement_of(obj)
        self.mov_norm[idx] = m
 


    def _movement_of(self, obj):
        """ Returns the degree of movement of the object. >= 1 means you will lose! """

        # PERFECT! DON'T TOUCH!
        FACTOR_LINEAR = 2.95696759223938 / 2.0 * .75
        FACTOR_ANGULAR = .75

        v1 = obj.linearVelocity.length  / FACTOR_LINEAR
        v2 = abs(obj.angularVelocity)   / FACTOR_ANGULAR

        v_max = max(v1, v2)

        return v_max  


    ### /movement detection

    def _compute_obs(self):
        return 1.0 if not self.doll_can_move else self._get_progress()

    def _compute_obs_speed(self):
        return float(self.doll_tick_duration) / float(max(DOLL_MOVE_MAX, DOLL_STILL_MAX))

    def _compute_phase_time(self):
        if self.doll_can_move:
            low  = DOLL_MOVE_MIN
            high = DOLL_MOVE_MAX 
        else:
            low  = DOLL_STILL_MIN
            high = DOLL_STILL_MAX 

        return random.randint(low, high)    


    def pre_render(self, env):
        env.color_sky = (1, 0, 0) if not self.doll_can_move else (0, 1-self._get_progress(), 0)

        if self.immunity > 0:
            env.hull.color1 = COLOR_KICKED
            for l in env.legs:
                l.color1 = COLOR_KICKED
        else:
            self._set_move_color(0, env.hull)
            self._set_move_color(1, env.legs[0])
            self._set_move_color(2, env.legs[1])
            self._set_move_color(3, env.legs[2])
            self._set_move_color(4, env.legs[3])



    def _set_move_color(self, idx, obj):
        val = self.mov_norm[idx]
        if val >= 1.0:
            c = (1.0, 1.0, 0.0)
        else: 
            c = (val, val, val) 
        
        obj.color1 = c



    def pre_step(self, env):
  
        self._do_movement_detection(env)
        
        # Shoot anyone that moves!
        self.motion_detected = False
        if self.doll_can_move == False and self.immunity <= 0:
            for m in self.mov_norm:
                if m >= 1.0:
                    self._doll_shoot(env)
                    # Grant immunity for a while
                    self.immunity = FPS * 10
                    self.motion_detected = True
                    break

        self.doll_tick_remaining -= 1
        if self.immunity > 0:
            self.immunity -= 1

        if self.doll_tick_remaining <= 0:
            self._doll_switch_phase()
        

    def post_reset(self, env):
        self.doll_can_move = False 
        self._doll_switch_phase() 
        env.game_over_on_hull_impact = True

        self.mov_norm = [0]    * 5   # normalized 

        self.immunity = 0
        self.last_x = 0
        self.motion_detected = False
      


    def _doll_switch_phase(self):
        self.doll_can_move = not self.doll_can_move
        self.doll_tick_remaining = self.doll_tick_duration = self._compute_phase_time()


    def _get_progress(self):
        a = self.doll_tick_duration - self.doll_tick_remaining
        return float(a) / float(self.doll_tick_duration)

    def _doll_shoot(self, env):
        if env.hull != None:
            env.hull.ApplyForceToCenter(DOLL_SHOOT_FORCE, True)


