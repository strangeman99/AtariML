from abc import ABC
import numpy as np
import gym
from gym import spaces
import time

class WalkingEnv(gym.Env, ABC):  # The custom environment for walking

    def __init__(self, motorNum, cogNum, maxCog):
        super(WalkingEnv, self).__init__()
        self.steps = 0
        self.timeNotFallen = 0
        self.startTime = time.time_ns()
        self.distance = 0.0
        self.motorNum = motorNum
        self.cogNum = cogNum
        self.maxCog = maxCog  # the maximum height of the centre of gravity

        motorHigh = np.array(self.motorNum, dtype=float)
        for obs in range(motorHigh.size):
            motorHigh[obs] = 360.0

        cogHigh = np.array(self.cogNum, dtype=float)
        for obs in range(cogHigh.size):
            cogHigh[obs] = self.maxCog

        motorLow = np.zeros(self.motorNum, dtype=float)
        cogLow = np.zeros(self.cogNum, dtype=float)

        # The observation space. I was thinking the motor positions and the centre of gravity of the robot
        self.observation_space = spaces.Dict({
            "motors": spaces.Box(low=motorLow, high=motorHigh),
            "cogs": spaces.Box(low=cogLow, high=cogHigh)
        })

        # Actions should be the position of each motor in the robot
        self.action_space = spaces.Box(low=motorLow, high=motorHigh)

    def step(self, action):  # What to do when it takes a step
        pass

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # I'm going to start the motor positions vertically
        self.steps = 0
        self.timeNotFallen = 0
        self.startTime = time.time_ns()
        self.distance = 0.0


        return observation
