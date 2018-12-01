import gym
import time

class OpenAI(object):
    def __init__(self):
        pass
    
    def hotterColder(self):
        env = gym.make('HotterColder-v0')

        try:
            tries = 50
            observation = env.reset()
            for t in range(tries):
                action = env.action_space.sample()
                
                observation, reward, done, info = env.step(action)
                print("Action: {} | Observation: {} | Reward: {}".format(action, observation, reward))
                
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
                
        except Exception as err:
            print("Error: "+str(err))
        finally:
            time.sleep(2)
            env.close()
