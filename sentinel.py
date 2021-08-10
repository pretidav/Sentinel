from numpy.core.fromnumeric import shape
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import gym
from gym import spaces
from gym.utils import seeding
import math
import cv2
#import pantilthat
import time

MAXDIST = np.sqrt(2)*255
class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.002)

    def create_model(self,layers=1):
        
        def rescale(a):
            return a*tf.constant(self.action_bound)

        state_input = tf.keras.layers.Input((self.state_dim,))
        dense = tf.keras.layers.Dense(5, activation='relu')(state_input)
        for l in range(1,layers-1):
            dense = tf.keras.layers.Dense(10, activation='relu')(dense)
        out_mu = tf.keras.layers.Dense(self.action_dim, activation='tanh')(dense)
        mu_output = tf.keras.layers.Lambda(lambda x: rescale(x))(out_mu)
        std_output = tf.keras.layers.Dense(self.action_dim, activation='softplus')(dense) 
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    def compute_loss(self, mu, std, actions, advantages):
        dist = tfp.distributions.Normal(loc=mu, scale=std)
        loss_policy = (-dist.log_prob(value=actions) * advantages + 0.002*dist.entropy())
        return tf.reduce_sum(loss_policy)
        
    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.002)

    def create_model(self):
        state_input = tf.keras.layers.Input((self.state_dim,))
        dense_1 = tf.keras.layers.Dense(5, activation='relu')(state_input)
        dense_2 = tf.keras.layers.Dense(10, activation='relu')(dense_1)
        v       = tf.keras.layers.Dense(1, activation='linear')(dense_2)
        return tf.keras.models.Model(state_input, v)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    @tf.function
    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = 2 
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-3, 1.0]
        self.gamma = 0.99
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

    def td_target(self, reward, next_state, done):
        if done:
            return reward
        v_value = self.critic.model.predict(
            np.reshape(next_state, [1, self.state_dim]))
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

class Tracker(gym.Env):
    def __init__(self,servo):
        self.servo = servo()

        self.min_action = -90.0
        self.max_action =  90.0

        self.min_position = 0.0
        self.max_position = 255.0
        self.goal_position = (
            0.0  
        )
        
        self.low_state = np.array(
            [self.min_position, self.min_position], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position], dtype=np.float32
        )

        self.low_action = np.array(
            [self.min_action, self.min_action], dtype=np.float32
        )
        self.high_action = np.array(
            [self.max_action, self.max_action], dtype=np.float32
        )


        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, shape=(2,), dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_rel_positions(self):
        #### writeme
        rel_x = 0
        rel_y = 0
        return rel_x, rel_y

    def step(self, action):
        
        rel_position_x = self.state[0]
        rel_position_y = self.state[1]

        new_theta = min(max(action[0], self.min_action), self.max_action)
        new_phi   = min(max(action[1], self.min_action), self.max_action)
        #pantilthat.pan(new_theta)
        #pantilthat.tilt(new_phi)

        tolerance = 0.1
        distance = np.sqrt(math.pow(rel_position_x[0], 2) + math.pow(rel_position_y[1], 2))
        done = bool(distance <= tolerance)

        reward = 0
        if done:
            reward = 10.0
        reward -= distance 

        rel_position_x, rel_position_y = self.servo.target()
        self.state = np.array([rel_position_x, rel_position_y])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-90.0, high=90.0), self.np_random.uniform(low=-90.0, high=90.0)])
        return np.array(self.state)


class Servo():
    def __init__(self):
        self.pan  = 0 
        self.tilt = 0
        self.pan, self.tilt = self.wake_up()

    def wake_up(self):
        pan = 0
        tilt = 0
        #pantilthat.pan(pan)
        #pantilthat.tilt(tilt)
        return pan, tilt

    def get_coordinates(self,
                frame,
                roi,
                panAngle=0,
                tiltAngle=0,
                frame_w=255,
                frame_h=255):

        (x, y, w, h) = tuple(map(int,roi))
        center_x = x+round(w/2)
        center_y = y+round(h/2)
        
        cv2.circle(frame,(int(center_x),int(center_y)), 3, (0,0,255), -1)
        cv2.circle(frame,(round(frame_w/2),round(frame_h/2)), 3, (255,0,0), -1)

        print('target_X, target_Y = {},{}'.format(center_x,center_y))
        print('center_X, center_Y = {},{}'.format(frame_w/2,frame_h/2))
        print('pan,tilt = {},{}'.format(panAngle,tiltAngle))
        return center_x-frame_w/2, center_y-frame_h/2

    def get_pan_tilt(self):
        pan  = 0 #pantilthat.get_pan()
        tilt = 0 #pantilthat.get_tilt()
        return pan, tilt

    def cam_show(self,frame):
        cv2.imshow('frame',frame)

    def target(self,flip=False):
        rel_x, rel_y = 0.0, 0.0
        pan, tilt = self.get_pan_tilt()
        cap = cv2.VideoCapture(-1)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(width,height)

        tracker      = cv2.legacy.TrackerMedianFlow_create()
        face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            
            roi   = face_cascade.detectMultiScale(frame,
                                                scaleFactor=1.2, 
                                                minNeighbors=5) 

            count = 0
            for (x, y, w, h) in roi: 
                tracker.init(frame,(x, y, w, h))
    
            ret, frame = cap.read()

            count += int(ret)
            
            success, roi = tracker.update(frame)
            (x,y,w,h) = tuple(map(int,roi))
            print(success)
            if success:
                p1 = (x, y)
                p2 = (x+w, y+h)
                cv2.rectangle(frame, p1, p2, (0,255,0), 3)
                rel_x, rel_y = self.get_coordinates(frame,roi,pan,tilt,width,height)
                
            if flip:
                frame = cv2.flip(frame, 0);
            self.cam_show(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        return rel_x, rel_y

def welcome():
    msg = '# RL Tracker - START #'
    print('#'*len(msg))
    print(msg)
    print('#'*len(msg))
    return time.time()

def bye(start_time):
    tot=time.time() - start_time
    msg='# RL Tracker - END - time:{} sec #'.format(tot)
    print('#'*len(msg))
    print(msg)
    print('#'*len(msg))


if __name__=='__main__':
    start_time = welcome()    


    Env   = Tracker(servo=Servo)
    agent = A2CAgent(env=Env)
    
    print(Env.action_space)
    print(Env.observation_space)
    print(Env.servo.target())



    bye(start_time=start_time)