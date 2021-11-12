import numpy as np
from time import time, sleep

from challenge_do_not_modify import BoatInUnknownWaters

# initialize environment:
env = BoatInUnknownWaters(n_steps=100)
parms = env.get_parameters()
print("parameters:", parms)

# a simple strategy:
def simple_strategy(obs):
    # turn boat until facing rather "up" than "down", 
    # then begin pushing forward and turn it further until facing straight "up"
    x, y, phi, dx_dt, dy_dt, dphi_dt = obs
    m = parms['m_max']
    rho = -np.sign(np.sin(phi)) * parms['rho_max'] * (np.abs(np.sin(phi)) if np.cos(phi) > 0 else 1)
    return [m, rho]

# run one episode:
print("\nRUNNING ONE EPISODE...")
obs = env.reset()
total = 0
while True:
    # choose next action by applying strategy to last observation:
    action = simple_strategy(obs)
    # let environment run for one step and get new observation and reward:
    obs, reward, terminated, info = res = env.step(action)
    total += reward
    # show state to user:
    env.render()
    sleep(0.1)
    # check whether episode ended:
    if terminated: 
        break

print('started with', env.history[0])
print('ended with', env.history[-1])
print('total reward:', total)

# now run many times without rendering to assess the strategy's performance:
n_episodes = 1000
print("\nRUNNING", n_episodes, "EPISODES...")
start_time = time()

n_success = 0
for episode in range(n_episodes):
    obs = env.reset()
    while True:
        action = simple_strategy(obs)
        obs, reward, terminated, info = res = env.step(action)
        total += reward
        if terminated: 
            break
    n_success += reward
rate = n_success / n_episodes
print("took", (time()-start_time)/n_episodes, "seconds per episode")
print("success rate", rate, "+-", np.sqrt(rate*(1-rate)/n_episodes))

sleep(10)
exit()
