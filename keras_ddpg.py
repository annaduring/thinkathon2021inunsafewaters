# adapted from https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py

# BEWARE: what is called "state" in this code is actually what is called "observation" in the environment.
# it includes both the actual state (x,y,phi) and its time derivative (dx/dt,dy/dt,dphi/dt)!

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

### our env specifics:
    
from challenge_do_not_modify import InUnsafeWaters, evaluate

problem = "InUnsafeWaters"
#env = InUnsafeWaters(n_steps=100, boundary="line")
env = InUnsafeWaters(n_steps=100, boundary="circle")

# CHOOSE A RANDOM SEED:
#   interesting cases in (rough subjective) order of ascending difficulty: 
#     for boundary=line: 7, 12, 37, 35, 23, 32
#     for boundary=circle: 39, 21, 37, 10, 38, 20
env.seed(37)

# choose whether to reuse th same scenario (flow):
# (must be false in the final evaluation!): 
reuse_scenario = True

# choose whether to use the real reward function (=survival yes or no, as used in final evaluation)
# or use survival time or squared survival time instead (may help in learning): 
#reward_function = 'real'
#reward_function = 'squared time'
reward_function = 'survival time'

# optionally weigh down some observation items:
obs_weights = np.ones(20)  # use all parts of the observation
#obs_weights = np.array([0,0,0,0, 1, 1,1, 0,0,0,0, 1, 1,1, 0,0,0,0,0,0])  # use D, theta and their derivs
#obs_weights = np.array([0,0,0,0, 0, 1,1, 0,0,0,0, 0, 0,0, 0,0,0,0,0,0])  # use only theta, as in "straight_away" strategy


# learner parameters:
    
total_episodes = 500  # JH: original: 1000
std_dev = 0.2  # JH: original: 0.2

# Discount factor for future rewards:
gamma = 0.99  # JH: maybe set to 1?

# Learning rate for actor-critic models:
critic_lr = 0.002
actor_lr = 0.001

# Used to update target networks:
tau = 0.005

###


num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.00, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer( minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)  # JH: replaced 1 by num_actions

    outputs = lower_bound + (outputs+1)/2 * (upper_bound-lower_bound)  # JH: fixed bounds!
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

#    return [np.squeeze(legal_action)]
    return np.squeeze(legal_action)  # JH: removed square brackets


ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()


# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

buffer = Buffer(50000, 64)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state = env.reset(same=reuse_scenario)
    episodic_reward = 0

    step = 0
    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        if any(np.isnan(prev_state)):
            print("WARNING, state contains nan values:", prev_state)

        action = policy(tf_prev_state, ou_noise)
        # Receive state and reward from environment.
        state, reward, done, info = env.step(action)
        # JH: optionally suppress or reweight part of the state:
        state *= obs_weights
        
        # JH: optionally use an auxiliary reward function instead:
        if reward_function == 'real':
            pass  # use the real rewards
        elif reward_function == 'survival time':
            reward = 3.0 / env.n_steps if not done else 0.0  
            # this gives 3.0 if surviving all of the n_steps many steps between time 0 and time 3 
        elif reward_function == 'squared time':
            reward = 18.0 * step / env.n_steps**2 if not done else 0.0
            # this gives 9.0 if surviving all of the n_steps many steps between time 0 and time 3 
            step += 1
        else:
            raise Exception('unknown reward function')

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# Save the weights
actor_model.save_weights("/tmp/actor.h5")
critic_model.save_weights("/tmp/critic.h5")

target_actor.save_weights("/tmp/target_actor.h5")
target_critic.save_weights("/tmp/target_critic.h5")


# JH: finally evaluate trained actor on actual reward function and show one run:

no_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.0) * np.ones(1))

def my_policy(obs):
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(obs), 0)
    return policy(tf_prev_state, no_noise)

# THIS IS THE LINE YOU NEED TO CALL BEFORE SUBMITTING YOUR SOLUTION AS WELL
# (with the seed value we tell you on Sunday morning):
rate = evaluate(my_policy, n_steps=100, seed=1)
# Then send us the output.

obs = env.reset(same=reuse_scenario)
while True:
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(obs), 0)
    action = policy(tf_prev_state, no_noise)
    print(action)
    obs, reward, terminated, info = res = env.step(action)
    env.render()
    sleep(0.1)
    if terminated: 
        break
sleep(100)

