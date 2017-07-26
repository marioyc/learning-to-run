import tensorflow as tf

session = tf.Session()

"""from osim.env import RunEnv

env = RunEnv(visualize=True)
observation = env.reset(difficulty = 0)

total_reward = 0
for i in range(200):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    total_reward += reward

print("Total reward %f" % total_reward)
"""
