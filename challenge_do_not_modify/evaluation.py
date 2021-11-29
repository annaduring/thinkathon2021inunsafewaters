from time import time, sleep
from progressbar import progressbar
from inspect import getsource
from hashlib import md5
import numpy as np

from . import InUnsafeWaters

def evaluate(policy, n_steps, seed):
    """Final evaluation of policies.
    
    policy: function(obs) -> action
    n_steps: how often you want your policy be called during the time from 0 to 3
    seed: the seed value provided by us on Sunday morning 
    """
    hash = md5(getsource(InUnsafeWaters).encode('utf-8')).hexdigest()
    assert hash == '81c921b2d666cd22ed223a4634a78615', "Wrong env version. Please use the original version provided! " + hash

    env = InUnsafeWaters(n_steps=n_steps, boundary='circle')
    env.seed(seed)

    # run 400 times to assess the strategy's performance:

    n_episodes = 400
    print("\nRUNNING FOR", n_episodes, "EPISODES...")

    start_time = time()   
    n_success = 0

    for episode in progressbar(range(n_episodes)):
        obs = env.reset(same=False)
        while True:
            action = policy(obs)
            obs, reward, terminated, info = res = env.step(action)
            if terminated: 
                break
        n_success += reward

    rate = n_success / n_episodes

    print("took", (time()-start_time)/n_episodes, "seconds per episode")  # should be below 1 sec

    # report success rate and its std.err.:
    print("SUCCESS RATE", rate, "+-", np.sqrt(rate*(1-rate)/n_episodes))

    return rate
    
