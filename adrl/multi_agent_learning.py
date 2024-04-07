import csv
from dacbench.benchmarks import SigmoidBenchmark
import pathlib


def make_multi_agent_env():
    bench = SigmoidBenchmark()
    bench.config.instance_set = {}
    with open(pathlib.Path(__file__).parent.resolve() / "sigmoid_train.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            f = []
            inst_id = None
            for i in range(len(row)):
                if i == 0:
                    try:
                        inst_id = int(row[i])
                    except Exception:
                        continue
                else:
                    try:
                        f.append(float(row[i]))
                    except Exception:
                        continue
            if not len(f) == 0:
                bench.config.instance_set[inst_id] = f

    bench.config.test_set = {}
    with open(pathlib.Path(__file__).parent.resolve() / "sigmoid_test.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            f = []
            inst_id = None
            for i in range(len(row)):
                if i == 0:
                    try:
                        inst_id = int(row[i])
                    except Exception:
                        continue
                else:
                    try:
                        f.append(float(row[i]))
                    except Exception:
                        continue
            if not len(f) == 0:
                bench.config.test_set[inst_id] = f

    bench.config["multi_agent"] = True
    env = bench.get_environment()
    return env


if __name__ == "__main__":
    env = make_multi_agent_env()

    # Add one agent per action dimension
    env.register_agent(agent_id=0)
    env.register_agent(agent_id=1)

    env.reset()
    total_reward = 0
    terminated, truncated = False, False
    while not (terminated or truncated):
        for agent in [0, 1]:
            observation, reward, terminated, truncated, info = env.last()
            action = env.action_spaces[agent].sample()
            env.step(action)
        observation, reward, terminated, truncated, info = env.last()
        total_reward += reward

    print(f"The final reward was {total_reward}.")
    env.close()
