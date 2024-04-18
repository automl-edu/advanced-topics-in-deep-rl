import numpy as np
from gymnasium import Wrapper
from carl.envs import CARLLunarLander


class GravityChangeWrapper(Wrapper):
    def __init__(self, env, gravity_change_interval=10000, gravity_change_kind=None):
        super().__init__(env)
        self.n_steps = 0
        self.n_total_steps = 0
        self.n_switches = 0
        self.gravity_change_interval = gravity_change_interval
        self.gravity_change_kind = gravity_change_kind

    def step(self, action):
        self.n_steps += 1
        state, reward, terminated, truncated, info = self.env.step(action)
        if self.n_steps >= self.gravity_change_interval:
            truncated = True
        return state, reward, terminated, truncated, info

    def reset(self):
        self.n_total_steps += self.n_steps
        self.n_steps = 0
        
        if self.n_total_steps  // self.gravity_change_interval > self.n_switches:
            change_kind = self.gravity_change_kind
            if self.gravity_change_kind is None:
                change_kind = np.random.choice(["flip", "random"])
            if change_kind == "flip":
                gravity = -self.env.context["GRAVITY_Y"]
            else:
                gravity = np.random.uniform(-10, 0)
            self.env.contexts[0] = {"GRAVITY_Y": gravity}
            self.env.context["GRAVITY_Y"] = gravity
            self.n_switches += 1
            self.env._update_context()
        return self.env.reset()


def make_continual_rl_env(gravity_change=None, gravity_change_interval=10000):
    contexts = {0: {"GRAVITY_Y": -10}}
    env = CARLLunarLander(contexts=contexts)
    env = GravityChangeWrapper(env, gravity_change_kind=gravity_change, gravity_change_interval=gravity_change_interval)
    return env


if __name__ == "__main__":
    print("")
    print("First, we flip the gravity.")
    env = make_continual_rl_env(gravity_change="flip")
    env.reset()
    for i in range(50000):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
        if i % 10000 == 0:
            print(f"Gravity is {env.env.unwrapped.world.gravity}")
    env.close()

    print("")
    print("Then we sample it randomly.")
    env = make_continual_rl_env(gravity_change="random")
    env.reset()
    for i in range(50000):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
        if i % 10000 == 0:
            print(f"Gravity is {env.env.unwrapped.world.gravity}")
    env.close()

    print("")
    print("And now we do both, but more often.")
    env = make_continual_rl_env(gravity_change_interval=5000)
    env.reset()
    for i in range(50000):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
        if i % 5000 == 0:
            print(f"Gravity is {env.env.unwrapped.world.gravity}")
    env.close()
