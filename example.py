import numpy as np
from time import sleep

from challenge_do_not_modify import BoatInUnknownWaters

# initialize environment:
env = BoatInUnknownWaters(n_steps=100)
parms = env.get_parameters()
print("parameters:", parms)

# a simple strategy:
def simple_strategy(obs):
    # turn boat until facing rather "up" than "down", 
    # then being pushing forward and turn it further until facing straight "up"
    x, y, phi, dx_dt, dy_dt, dphi_dt = obs
    m = parms['m_max']
    rho = -np.sign(np.sin(phi)) * parms['rho_max'] * (np.abs(np.sin(phi)) if np.cos(phi) > 0 else 1)
    return [m, rho]

# run one episode:
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

print('total reward:', total)
sleep(10)
