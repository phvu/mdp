from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


class MDP(object):

    def __init__(self, n_states, n_actions, rewards, transitions, terminals=None, gamma=0.9):
        """
        Construct an MDP

        :param n_states: number of states
        :param n_actions: number of actions
        :param rewards: array of size n_states, immediate reward of every state
        :param terminals: a list of bool [True, False...] of length n_states, denoting
            which states are terminals.
            For terminal states, Utility of those states will be its rewards.
        :param transitions: a list of list [state, action, next_state, probability]
            will be converted into a dict of (state, action) -> {next_state_1: prob1, next_state_2: prob2}
        :param gamma: discount factor
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.rewards = rewards
        self.transitions = {}
        for s, a, s1, p in transitions:
            if (s, a) not in self.transitions:
                self.transitions[(s, a)] = {}
            self.transitions[(s, a)][s1] = p
        self.gamma = gamma
        self.terminals = terminals

    def reward(self, state_idx):
        return self.rewards[state_idx]

    def transition(self, state_idx, action_idx):
        return self.transitions.get((state_idx, action_idx), {})

    def _get_utility(self, utilities, state_idx):
        return self.reward(state_idx) if self.terminals[state_idx] else utilities[state_idx]

    def value_iteration(self, utilities=None, n_iters=10, verbose=False):
        if utilities is None:
            utilities = [0.0] * self.n_states
        elif len(utilities) != self.n_states:
            raise ValueError('utilities has to have {} elements'.format(self.n_states))

        for i in range(n_iters):
            new_utilities = [0] * self.n_states
            for s in range(self.n_states):
                if self.terminals[s]:
                    new_utilities[s] = self.reward(s)
                else:
                    max_neighbor_util = None
                    for a in range(self.n_actions):
                        util = sum(prob * self._get_utility(utilities, next_state)
                                   for next_state, prob in self.transition(s, a).items())
                        max_neighbor_util = util if max_neighbor_util is None else max(max_neighbor_util, util)

                    new_utilities[s] = (self.reward(s) +
                                        self.gamma * (0 if max_neighbor_util is None else max_neighbor_util))
            utilities = new_utilities

            if verbose:
                print('Iteration #{}: {}'.format(i, utilities))

        return utilities

    def policy_iteration(self, policy=None, n_iters=10, verbose=False):
        if policy is None:
            policy = np.random.randint(0, self.n_actions, self.n_states).tolist()
        elif len(policy) != self.n_states:
            raise ValueError('policy has to have {} elements'.format(self.n_states))

        utilities = [0] * self.n_states

        for i in range(n_iters):

            new_utilities = [0] * self.n_states
            # calculate current utility
            for s in range(self.n_states):
                if self.terminals[s]:
                    new_utilities[s] = self.reward(s)
                else:
                    neighbor_util = sum(prob * self._get_utility(utilities, next_state)
                                        for (next_state, prob) in self.transition(s, policy[s]).items())
                    new_utilities[s] = self.reward(s) + self.gamma * neighbor_util
            utilities = new_utilities

            # improve current policy
            for s in range(self.n_states):
                max_exp_util = None
                best_action = -1
                for a in range(self.n_actions):
                    util = sum(prob * utilities[next_state] for next_state, prob in self.transition(s, a).items())
                    if max_exp_util is None or max_exp_util < util:
                        max_exp_util = util
                        best_action = a
                policy[s] = best_action

            if verbose:
                print('Iteration #{}: {}'.format(i, policy))
        return policy
