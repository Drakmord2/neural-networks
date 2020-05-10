import gym
import time
import pandas as pd
import numpy as np


class OpenAI(object):
    def __init__(self, agent=None):
        self.agent = agent
        self.action_space = None
        self.observations = []
        self.rewards = []

    def hotterColder(self):
        try:
            env = gym.make('HotterColder-v0')
            tries = 50
            observation = env.reset()
            for t in range(tries):
                action = env.action_space.sample()

                observation, reward, done, info = env.step(action)
                print("Action: {} | Observation: {} | Reward: {}".format(action, observation, reward))

                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break

        except Exception as err:
            print("- Error: {}".format(err))
        finally:
            env.close()

    def cartPole(self):
        try:
            env = gym.make('CartPole-v1')

            epochs = 10
            for e in range(epochs):
                print("Epoch: " + str(e))

                observation = env.reset()
                while True:
                    env.render()

                    example = pd.DataFrame(list(observation))
                    value = self.agent.compute(example)

                    self.make_action("Box", value)
                    action = self.action_space.sample()
                    action = int(action[0])

                    print("- Action: " + str(action))

                    observation, reward, done, info = env.step(action)

                    if done:
                        time.sleep(2)
                        break

        except Exception as err:
            print("- Error: {}".format(err))
            raise err
        finally:
            env.close()

    # -----------------------------------------------------------------------------
    # Auxiliary
    # -----------------------------------------------------------------------------

    def generate_data(self, env):
        data = []
        MIN_SCORE = 70
        trainings = 1000

        for i in range(trainings):
            train = []
            score = 0
            tries = 250
            observation = env.reset()
            for t in range(tries):
                action = env.action_space.sample()

                observation, reward, done, info = env.step(action)

                score += reward
                observation = list(observation)
                observation.append(action)

                if not done:
                    train.append(observation)

                if done:
                    if score >= MIN_SCORE:
                        print("Score: {}".format(score))
                        data += train
                        train = []
                    break

        datadf = pd.DataFrame(data, columns=['obs1', 'obs2', 'obs3', 'obs4', 'action'])
        datadf.to_csv("data/cartpole_data.csv", index=False)

        print("- Generated Training Data -\n")
        print("Data: " + str(datadf.tail()))

    def get_data(self, outputs):
        data = []
        for o in range(len(self.observations)):
            result = []
            for r in range(outputs):
                rew = self.rewards[o][0] if isinstance(self.rewards[o], list) else self.rewards[o]
                result.append(rew)

            data.append(list(self.observations[0]) + result)

        data = pd.DataFrame(data)
        return data

    def make_action(self, atype, value):
        if atype == "Box":
            self.action_space = gym.spaces.Box(np.array(value), np.array(value), dtype=np.float64)
