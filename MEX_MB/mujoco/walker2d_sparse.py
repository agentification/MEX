import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dSparseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'walker2d.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        forward_vel = (posafter - posbefore) / self.dt
        self._goal_vel = 1.0
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        sparse_reward = self.sparsify_rewards(forward_reward)
        ctrl_cost =1e-3 * np.sum(np.square(a))
        reward = sparse_reward - ctrl_cost
        s = self.state_vector()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}
    
    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        self.goal_radius = 0.5
        if r < - self.goal_radius:
            r = -2
        r = r + 2
        return r

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
    

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
