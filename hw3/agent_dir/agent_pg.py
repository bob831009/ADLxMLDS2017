from agent_dir.agent import Agent
import scipy
import numpy as np
import tensorflow as tf
import multiprocessing
import gym
import threading
import os
from scipy.misc import imresize


def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def pipeline(image, new_HW=(80, 80), height_range=(35, 193), bg=(144, 72, 17)):
    """Returns a preprocessed image
    (1) Crop image (top and bottom)
    (2) Remove background & grayscale
    (3) Reszie to smaller image
    Args:
        image (3-D array): (H, W, C)
        new_HW (tuple): New image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color (R, G, B)
    Returns:
        image (3-D array): (H, W, 1)
    """
    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)
    image = np.expand_dims(image, axis=2)

    return image


def resize_image(image, new_HW):
    """Returns a resized image
    Args:
        image (3-D array): Numpy array (H, W, C)
        new_HW (tuple): Target size (height, width)
    Returns:
        image (3-D array): Resized image (height, width, C)
    """
    return imresize(image, new_HW, interp="nearest")


def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom
    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept
    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):
    """Make the background 0
    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)
    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image


def discount_reward(rewards, gamma=0.99):
    """Returns discounted rewards
    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate
    Returns:
        discounted_rewards: same shape as `rewards`
    Notes:
        In Pong, when the reward can be {-1, 0, 1}.
        However, when the reward is either -1 or 1,
        it means the game has been reset.
        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim, logdir=None):
        """Network structure is defined here
        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

            with tf.variable_scope("layer1"):
                net = tf.layers.conv2d(net,
                                       filters=16,
                                       kernel_size=(8, 8),
                                       strides=(4, 4),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("layer2"):
                net = tf.layers.conv2d(net,
                                       filters=32,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       name="conv")
                net = tf.nn.relu(net, name="relu")

            with tf.variable_scope("fc1"):
                net = tf.contrib.layers.flatten(net)
                net = tf.layers.dense(net, 256, name='dense')
                net = tf.nn.relu(net, name='relu')

            # actor network
            actions = tf.layers.dense(net, output_dim, name="final_fc")
            self.action_prob = tf.nn.softmax(actions, name="action_prob")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            self.actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards,
                                                           predictions=self.values)

            self.total_loss = self.actor_loss + self.value_loss * .5
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=.99)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)


class sub_Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, logdir=None):
        """sub_Agent worker thread
        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.env): Gym environment
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork (H, W, C)
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary
                TODO: Add summary
        """
        super(sub_Agent, self).__init__()
        self.local = A3CNetwork(name, input_shape, output_dim, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir
        self.reward_log_file = open('./logs/%s.txt' % self.name, 'a')
        self.play_episode_num = 0
        self.cache_reward = []

    def print(self, reward):
        message = "Agent(name={}, episode={}, reward={})".format(self.name, self.play_episode_num, reward)
        print(message)
        self.reward_log_file.write('%d\n' % (reward))
        self.cache_reward.append(reward)
        if(len(self.cache_reward) >= 10):
            print('Agent %s mean reward: %lf' % (self.name, sum(self.cache_reward) / float(len(self.cache_reward))))
            self.cache_reward = []


    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []

        s = self.env.reset()
        s = pipeline(s)
        state_diff = s

        done = False
        total_reward = 0
        time_step = 0
        while not done:

            a = self.choose_action(state_diff)
            s2, r, done, _ = self.env.step(a)

            s2 = pipeline(s2)
            total_reward += r

            states.append(state_diff)
            actions.append(a)
            rewards.append(r)

            state_diff = s2 - s
            s = s2

            if r == -1 or r == 1 or done:
                time_step += 1

                if time_step >= 5 or done:
                    self.train(states, actions, rewards)
                    self.sess.run(self.global_to_local)
                    states, actions, rewards = [], [], []
                    time_step = 0

        self.play_episode_num += 1
        self.print(total_reward)

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def choose_action(self, states):
        """
        Args:
            states (2-D array): (N, H, W, 1)
        """
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {
            self.local.states: states
        }
        # print(states)
        action = self.sess.run(self.local.action_prob, feed)
        # print(action)
        action = np.squeeze(action)
        # print(action)

        return np.random.choice(np.arange(self.output_dim) + 1, p=action)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions) - 1
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = discount_reward(rewards, gamma=0.99)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards) + 1e-8

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)



class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.coord = tf.train.Coordinator()

        self.save_path = "./models/A3C/model.ckpt"
        self.n_threads = 12
        self.input_shape = [80, 80, 1]
        self.output_dim = 3  # {1, 2, 3}
        self.global_network = A3CNetwork(name="global",
                                    input_shape=self.input_shape,
                                    output_dim=self.output_dim)


        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            if tf.train.get_checkpoint_state(os.path.dirname(self.save_path)):
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(self.sess, self.save_path)
                print("Model restored to global")
            else:
                print("No model is found")
                exit(0)


        self.env = env

        ##################
        # YOUR CODE HERE #
        ##################

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.past_observation = pipeline(self.env.reset())


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        try:
            thread_list = []
            env_list = []

            for id in range(self.n_threads):
                env = gym.make("Pong-v0")

                if id == 0:
                    env = gym.wrappers.Monitor(env, "monitors", force=True)

                single_agent = sub_Agent(env=env,
                                     session=self.sess,
                                     coord=self.coord,
                                     name="thread_{}".format(id),
                                     global_network=self.global_network,
                                     input_shape=self.input_shape,
                                     output_dim=self.output_dim)
                thread_list.append(single_agent)
                env_list.append(env)

            if tf.train.get_checkpoint_state(os.path.dirname(self.save_path)):
                var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(self.sess, self.save_path)
                print("Model restored to global")
            else:
                init = tf.global_variables_initializer()
                self.sess.run(init)
                print("No model is found")

            for t in thread_list:
                t.start()

            print("Ctrl + C to close")
            self.coord.wait_for_stop()

        except KeyboardInterrupt:
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            saver.save(self.sess, self.save_path)
            print()
            print("=" * 10)
            print('Checkpoint Saved to {}'.format(self.save_path))
            print("=" * 10)

            print("Closing threads")
            self.coord.request_stop()
            self.coord.join(thread_list)

            print("Closing environments")
            for env in env_list:
                env.close()

            self.sess.close()


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = pipeline(observation)
        states = observation - self.past_observation
        self.past_observation = observation
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {self.global_network.states: states}

        action = self.sess.run(self.global_network.action_prob, feed)
        action = np.squeeze(action)
        # print(action)
        return np.argmax(action) + 1
        # return np.random.choice(np.arange(self.output_dim) + 1, p=action)

