import numpy as np
from gymnasium import Wrapper
from carl.envs import CARLLunarLander


class GravityChangeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_steps = 0
        self.n_total_steps = 0
        self.n_switches = 0

    def step(self, action):
        self.n_steps += 1
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.n_steps >= 10000:
            truncated = True
        return state, reward, terminated, truncated, info

def reset(self):
        self.n_total_steps += self.n_steps
        self.n_steps = 0
        
        if self.n_total_steps  // 10000 > self.n_switches:
            # as gravity has to be in (-20. 0.01) flipping does not make sense
            # change_kind = np.random.choice(["flip", "random"])
            change_kind = "random"  
            
            if change_kind == "flip":
                gravity = -self.env.context["GRAVITY_Y"]
            else:
                gravity = np.random.uniform(-20, 0)
            self.env.contexts[0] = {"GRAVITY_Y": gravity}
            self.env.context["GRAVITY_Y"] = gravity
            self.n_switches += 1
        return self.env.reset()


def make_continual_rl_env():
    contexts = {0: {"GRAVITY_Y": -10}}
    env = CARLLunarLander(contexts=contexts)
    env = GravityChangeWrapper(env)
    return env


if __name__ == "__main__":
    env = make_continual_rl_env()
    env.reset()
    for i in range(50000):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
        if env.n_steps % 10000 == 0:
            print(f"Gravity is {env.env.context['GRAVITY_Y']}")
    env.close()
