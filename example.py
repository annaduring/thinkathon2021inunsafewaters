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
env.seed(10)

parms = env.get_parameters()
print("parameters:", parms)


# some simple strategies:
def random_action(obs):
    # choose random motor speed and rudder angle
    m = env.np_random.uniform(0, parms['m_max'])
    rho = env.np_random.uniform(-parms['rho_max'], parms['rho_max'])
    return [m, rho]   
def straight_away(obs):
    # turn boat until facing rather "away" than "towards" boundary, 
    # then begin pushing forward and turn it further until facing straight away from boundary
    x, y, sinphi, cosphi, D, sintheta, costheta, dx, dy, dsinphi, dcosphi, dD, dsintheta, dcostheta, fx, fy, dxfx, dyfx, dxfy, dyfy = obs
    # so we want theta = pi, i.e. cos(theta) = -1 and sin(theta) = 0:
    m = parms['m_max']
    rho = -np.sign(sintheta) * parms['rho_max'] * (np.abs(sintheta) if costheta < 0 else 1)
    return [m, rho]
    
    
# CHOOSE A STRATEGY TO TEST:
#my_strategy = random_action
my_strategy = straight_away


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
