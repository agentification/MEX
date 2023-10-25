import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahSparseEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.sim.data.qpos[0]

        forward_vel = (posafter - posbefore) / self.dt
        self._goal_vel = 1.5  #1.5
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        sparse_reward = self.sparsify_rewards(forward_reward)
        ctrl_cost =  1e-1 * np.sum(np.square(a))
        reward = sparse_reward - ctrl_cost
        s = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(reward_run=sparse_reward, reward_ctrl=-ctrl_cost)
    
    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        self.goal_radius = 0.6
        if r < - self.goal_radius:
            r = -2
        r = r + 2
        return r
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
