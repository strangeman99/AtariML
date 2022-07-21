from abc import ABC

import gym

class WalkingEnv(gym.Env, ABC):  # The custom environment for walking

    def __init__(self):
        # The observation space. I was thinking the motor positions and the centre of gravity of the robot
        #  self.observation_space =

        # Actions should be the position of each motor in the robot
        #  self.action_space =
        pass


    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # I'm going to start the motor positions vertically
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2)

        observation = self.observation_space
        return observation
