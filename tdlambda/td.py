from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from collections import namedtuple

Move = namedtuple('State', ['state', 'reward'])


class TDLambda(object):

    def __init__(self, state_count):
        self.state_count = state_count

    def run(self, episodes, gamma=0.1, lambda_val=1, learning_rate=lambda t: 1.0/t):
        """
        Run the TD(lambda) algorithm

        :param episodes: list[list[Move]]
        :param gamma: gamma of the MDP model (discounted reward factor)
        :param lambda_val: value of lambda in TD(lambda)
        :param learning_rate: Learning rate function
        :return: list of values of the states
        """
        v_old = [0] * self.state_count
        for t, episode in enumerate(episodes):
            e = [0] * self.state_count
            v = v_old[:]
            for j, s in enumerate(episode[:-1]):
                e[s.state] += 1

                for i in range(0, self.state_count):
                    diff = s.reward + (gamma * v_old[episode[j+1].state]) - v_old[s.state]
                    v[i] += learning_rate(i + 1) * e[i] * diff
                    e[i] *= lambda_val * gamma
            v_old = v[:]
        return v_old
