import numpy as np
from time import time, sleep
from progressbar import progressbar

from gym.wrappers import RecordVideo
from challenge_do_not_modify import InUnsafeWaters

# initialize environment:

# CHOOSE A BOUNDARY GEOMETRY:
#env = InUnsafeWaters(n_steps=100, boundary='line')  # simpler
env = InUnsafeWaters(n_steps=100, boundary='circle')  # harder

# CHOOSE A RANDOM SEED:
#   interesting cases in (rough subjective) order of ascending difficulty: 
#     for boundary=line: 7, 12, 37, 35, 23, 32
#     for boundary=circle: 39, 21, 37, 10, 38, 20
env.seed(39)

parms = env.get_parameters()
print("parameters:", parms)


# some simple strategies:
def random_action(obs):
    # choose random motor speed and rudder angle
    m = env.np_random.uniform(0, parms['m_max'])
    rho = env.np_random.uniform(-parms['rho_max'], parms['rho_max'])
    return [m, rho]   
def go_north(obs):
    # turn boat until facing rather "up" (=north) than "down" (south), 
    # then begin pushing forward and turn it further until facing straight "up"
    x, y, phi, dx_dt, dy_dt, dphi_dt = obs
    m = parms['m_max']
    rho = -np.sign(np.sin(phi)) * parms['rho_max'] * (np.abs(np.sin(phi)) if np.cos(phi) > 0 else 1)
    return [m, rho]
def go_center(obs):
    # turn boat until facing rather "inwards" (=to the center) than "outwards", 
    # then begin pushing forward and turn it further until facing straigth to the center
    x, y, phi, dx_dt, dy_dt, dphi_dt = obs
    m = parms['m_max']
    target_phi = np.arctan2(x,y) + np.pi
    rho = -np.sign(np.sin(phi-target_phi)) * parms['rho_max'] * (np.abs(np.sin(phi-target_phi)) if np.cos(phi-target_phi) > 0 else 1)
    return [m, rho]
    
# CHOOSE A STRATEGY TO TEST:
#my_strategy = random_action
#my_strategy = go_north
my_strategy = go_center


print("\nVIDEO RECORDING ONE EPISODE...")
# prepare video recording:
recording_env = RecordVideo(env, 
                            video_folder="/tmp/boat_videos", 
                            name_prefix='video_'+env.boundary+'_'+str(env._seed))
# first reset the environment, then (!) start the recorder:
obs = recording_env.reset()
recording_env.start_video_recorder()
total = 0
while True:
    # choose next action by applying strategy to last observation:
    action = my_strategy(obs) 
    # let environment run for one step and get new observation and reward:
    obs, reward, terminated, info = res = recording_env.step(action)
    total += reward
    sleep(0.1)
    # check whether episode ended:
    if terminated: 
        break
recording_env.close_video_recorder()


print('started with', env.history[0])
print('ended with', env.history[-1])
print('total reward:', total)


# now run many times without rendering to assess the strategy's performance:
n_episodes = 100
print("\nRUNNING", n_episodes, "EPISODES...")
start_time = time()

n_success = 0
for episode in progressbar(range(n_episodes)):
    obs = env.reset()
    while True:
        action = my_strategy(obs)
        obs, reward, terminated, info = res = env.step(action)
        if terminated: 
            break
    n_success += reward
rate = n_success / n_episodes
print("took", (time()-start_time)/n_episodes, "seconds per episode")
print("success rate", rate, "+-", np.sqrt(rate*(1-rate)/n_episodes))

sleep(10)
exit()
