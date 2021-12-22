import gym

class RandomFirstMove(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        super(RandomFirstMove, self).__init__(env)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        while True:
            self.env.reset()
            action = self.env.action_space.sample()

            state, _, done, _ = self.env.step(action)

            if not done:
                return state