from agent_dir.agent import Agent

import os
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, in_channels=4, num_actions=4, duel_net=False):
        super(DQN, self).__init__()
        self.duel_net = duel_net

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        if(self.duel_net):
            self.fc_value = nn.Linear(512, 1)
            self.fc_advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.leaky_relu(self.fc4(x.view(x.size(0), -1)))
        if(self.duel_net):
            value = self.fc_value(x)
            advantange = self.fc_advantage(x)
            q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
            return q

        else:
            return self.fc5(x)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        self.env = env
        # Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
        self.VALID_ACTIONS = [0, 1, 2, 3]
        

        ##################
        # YOUR CODE HERE #
        ##################
        self.model_dir = './models/DQN'
        self.model_path = os.path.join(self.model_dir, 'model')

        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000000
        self.num_episodes = 3000000
        self.memory_size = 10000
        self.learning_start_step = 10000
        self.replace_target_iter=1000
        self.num_actions = 4
        self.lr = 0.0001
        self.load_model = False
        # for bonus setting
        self.double_q = False
        self.duel_net = False
        self.epsilons = np.linspace(self.EPS_START, self.EPS_END, num=self.EPS_DECAY)

        self.Q_model = DQN(
            num_actions=self.num_actions, duel_net=self.duel_net)
        self.target_Q_model = DQN(
            num_actions=self.num_actions, duel_net=self.duel_net)

        if use_cuda:
            self.Q_model.cuda()
            self.target_Q_model.cuda()

        self.optimizer = optim.RMSprop(self.Q_model.parameters(), lr = self.lr)
        self.memory = ReplayMemory(self.memory_size)
        self.loss_func = nn.MSELoss()
        self.steps_done = 0
        self.f_reward = open(os.path.join(self.model_dir, 'reward_for_plot.txt'), 'a')

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.Q_model.load_state_dict(torch.load(self.model_path))
            random.seed(831009)

        if self.load_model and not args.test_dqn:
            print('loading trained model')
            self.Q_model.load_state_dict(torch.load(self.model_path))
            self.EPS_START = 0.05



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        last30_reward = np.zeros(30)
        for i_episode in range(self.num_episodes):
            # save model
            if(i_episode % 100 == 0 and i_episode != 0):
                print('save current model...')
                torch.save(self.Q_model.state_dict(), self.model_path)

            # Initialize the environment and state
            observation = self.env.reset()

            epi_reward = 0
            epi_step = 0
            for t in count():
                state = self.Observ2Tensor(np.transpose(observation, (2, 0, 1)))
                # Select and perform an action
                action = self.select_action(state)
                observation_, reward, done, _ = self.env.step(action[0, 0])
                state_ = self.Observ2Tensor(np.transpose(observation_, (2, 0, 1)))

                if done:
                    state_ = None

                epi_reward += reward
                reward = Tensor([reward])

                # Store the transition in memory
                self.memory.push(state, action, state_, reward)

                # swap observation
                observation = observation_

                # Perform one step of the optimization (on the target network)
                if (self.steps_done > self.learning_start_step) and (self.steps_done % 4 == 0):
                    self.optimize_model()

                if done:
                    break
                epi_step +=1
            last30_reward[i_episode%30] = epi_reward
            print('step: %d, epi: %d, reward_mean: %lf, reward: %d, epi_step: %d' % (self.steps_done, i_episode, np.mean(last30_reward), epi_reward, epi_step))
            self.f_reward.write('%d\n' % int(epi_reward))

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = self.Observ2Tensor(np.transpose(observation, (2, 0, 1)))
        action = self.select_action(state, test=True)
        return action[0, 0]

    def Observ2Tensor(self, observation):
        return torch.from_numpy(observation).unsqueeze(0).type(Tensor)

    def select_action(self, state, test=False):
        sample = random.random()
        # eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        #     math.exp(-1. * self.steps_done / self.EPS_DECAY)
        eps_threshold = self.epsilons[min(self.steps_done, self.EPS_DECAY-1)]
        
        self.steps_done += 1
        if(test): eps_threshold = 0.001

        if sample > eps_threshold:
            return self.Q_model(
                Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        else:
            return LongTensor([[random.randrange(self.num_actions)]])

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        if self.steps_done % self.replace_target_iter == 0:
            self.target_Q_model.load_state_dict(self.Q_model.state_dict())

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.Q_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE).type(Tensor))

        q_next = self.target_Q_model(non_final_next_states).detach()
        if(self.double_q):
            q_eval4next = self.Q_model(non_final_next_states).detach().max(1)[1]
            q_next = q_next.gather(1, q_eval4next.unsqueeze(1)).squeeze(1)
        else:
            q_next = q_next.max(1)[0]

        next_state_values[non_final_mask] = q_next
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.loss_func(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # ===== for test =====
        # for param in self.Q_model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

