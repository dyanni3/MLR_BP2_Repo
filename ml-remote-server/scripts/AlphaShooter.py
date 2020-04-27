"""
To Do:
1) on ue4 side output done in addition to state (and info)
2) be able to programmatically start and stop game
3) save rewards and progress to files for plotting and future use
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


#global variables
THIS_EPISODE_REWARD = 0
EPISODE_NUMBER = 0
STATE_DIMENSION = 6
ACTION_DIMENSION = 6
GAMMA = 0.99
N_EPISODES = 100
BATCH_SIZE = 64
HIDDEN_DIM = 12
CAPACITY = 50000
MAX_EPISODE = 50
MIN_EPS = 0.01

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
        self.old_state

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


def play_step(   agent: Agent,
                 replay_memory: ReplayMemory,
                 eps: float,
                 batch_size: int,
                 step: tuple,
                 this_episode_reward: float) -> int:

    s = agent.old_state
    if done:
    	r = -1
    if not done:
        a = agent.get_action(s, eps)
        s2, r, done, info = step
        total_reward += r
    replay_memory.push(s, a, r, s2, done)
    if len(replay_memory) > batch_size:
        minibatch = replay_memory.pop(batch_size)
        train_helper(agent, minibatch, GAMMA)
    agent.old_state = s2
    return (this_episode_reward, a)

def get_state_from_ue4(json_input):
	state = json_input['state']
	reward = json_input['reward']
	done = json_input['done']
	info = json_input['info']
	return(state, reward, done, info)

def epsilon_annealing(epsiode: int, max_episode: int, min_eps: float) -> float:
    slope = (min_eps - 1.0) / max_episode
    return max(slope * epsiode + 1.0, min_eps)

#MLPluginAPI
class AlphaShooterAPI(MLPluginAPI):

	#optional api: setup your model for training
	def on_setup(self):
		print('AlphaShooter setup running...')
		ue.log('AlphaShooter setup running...')
		agent = Agent(STATE_DIMENSION, ACTION_DIMENSION, HIDDEN_DIM)
		agent.old_state = 0
        replay_memory = ReplayMemory(CAPACITY)
        THIS_EPISODE_REWARD = 0
		
		
	#optional api: parse input object and return a result object, which will be converted to json for UE4
	def on_json_input(self, input_):
		print('state input received')
		print(f"python side: {input_}")
		ue.log(input_)
		step = get_state_from_ue4(input_)
		THIS_EPISODE_REWARD, action = play_step(agent,
			replay_memory,
			epsilon_annealing(EPISODE_NUMBER, MAX_EPISODE, MIN_EPS),
			BATCH_SIZE,
			step,
			THIS_EPISODE_REWARD)
		if input_['done'] == True:
			print("[Episode: {:5}] Reward: {:5} 𝜺-greedy: {:5.2f}".format(EPISODE_NUMBER + 1, THIS_EPISODE_REWARD, eps))
			EPISODE_NUMBER += 1
			THIS_EPISODE_REWARD = 0
			#Restart the game anew!
			#need to figure out how to do this from python

		ret_val = {"Name":"Current Action", "ActionValues":[action]}
		ret_val = json.dumps(ret_val)
		return (ret_val)

def get_api():
	#return CLASSNAME.get_instance()
	return AlphaShooterAPI.get_instance()