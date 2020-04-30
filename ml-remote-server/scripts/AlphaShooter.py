"""
To Do:
2) David - on the ue4 side fix 'done' to be on a timer
        and also on death done=true
        and then extract it from the json and deliver it
        to Nik as a python variable
3) save rewards and progress to files for plotting and future use
4) Nik - add A3C
5) David - add SAC
"""

print('AlphaShooter imports running')


import json
import torch
import random
import torch.nn
import numpy as np
import unreal_engine as ue
from collections import deque
from typing import List, Tuple
from collections import namedtuple
from mlpluginapi import MLPluginAPI

print("Imports success")


#global variables
class RL_Config():
	def __init__(self):
		self.THIS_EPISODE_REWARD = 0
		self.EPISODE_NUMBER = 0
		self.STATE_DIMENSION = 6
		self.ACTION_DIMENSION = 6
		self.GAMMA = 0.99
		self.N_EPISODES = 100
		self.BATCH_SIZE = 64
		self.HIDDEN_DIM = 12
		self.CAPACITY = 50000
		self.MAX_EPISODE = 50
		self.MIN_EPS = 0.01
		self.THIS_EPISODE_REWARD = 0


class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done"])


class ReplayMemory(object):

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cursor = 0
        self.memory = []

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool) -> None:
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class Agent(object):

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        self.dqn = DQN(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())
        self.old_state = 0
        self.done = False

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.choice(self.output_dim)
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data, 1)
            #need to decide whether to do one action at a time
            return int(argmax.numpy())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.dqn.train(mode=False)
        return self.dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        return loss


def train_helper(agent: Agent, minibatch: List[Transition], gamma: float) -> float:
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().data.numpy()
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * np.max(agent.get_Q(next_states).data.numpy(), axis=1) * ~done
    Q_target = agent._to_variable(Q_target)

    return agent.train(Q_predict, Q_target)


def play_step(done: bool,
	agent: Agent,
	replay_memory: ReplayMemory,
	eps: float,
	batch_size: int,
	step: tuple) -> int:
	
	s = agent.old_state
	if done:
		r = -1
	if not done:
		a = agent.get_action(s, eps)
		s2, r, done, info = step
		g.THIS_EPISODE_REWARD += r
		agent.done = done
	replay_memory.push(s, a, r, s2, done)
	if len(replay_memory) > batch_size:
		minibatch = replay_memory.pop(batch_size)
		train_helper(agent, minibatch, g.GAMMA)
	agent.old_state = s2
	return (a)

def get_state_from_ue4(json_input):
	state = json_input['state']
	reward = json_input['reward']
	done = json_input['done']
	info = json_input['info']
	return(state, reward, done, info)

def epsilon_annealing():
    slope = (g.MIN_EPS - 1.0) / g.MAX_EPISODE
    return max(slope * g.EPISODE_NUMBER + 1.0, g.MIN_EPS)

g = RL_Config()
agent = Agent(g.STATE_DIMENSION, g.ACTION_DIMENSION, g.HIDDEN_DIM)
replay_memory = ReplayMemory(g.CAPACITY)

#MLPluginAPI
class AlphaShooterAPI(MLPluginAPI):

	#optional api: setup your model for training
	def on_setup(self):
		print('AlphaShooter setup running...')
		ue.log('AlphaShooter setup running...')
		agent.done = False
		agent.old_state = 0
		
		
	#optional api: parse input object and return a result object, which will be converted to json for UE4
	def on_json_input(self, input_):
		print('state input received')
		print(f"python side: {input_}")
		ue.log(input_)
		step = get_state_from_ue4(input_)
		epsilon = epsilon_annealing()
		action = play_step(agent.done,
			agent,
			replay_memory,
			epsilon,
			g.BATCH_SIZE,
			step)
		if input_['done'] == True:
			print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(g.EPISODE_NUMBER + 1, THIS_EPISODE_REWARD, eps))
			g.EPISODE_NUMBER += 1
			g.THIS_EPISODE_REWARD = 0
			#Restart the game anew!
			#need to figure out how to do this from python

		ret_val = {"Name":"Current Action", "ActionValues":[action]}
		ret_val = json.dumps(ret_val)
		return (ret_val)

def get_api():
	#return CLASSNAME.get_instance()
	return AlphaShooterAPI.get_instance()