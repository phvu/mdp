from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from mdp.mdp import MDP

"""
_____________
|   |   | 1 |
|-----------|
|XXX|   |-1 |
|-----------|

5 states, starting from top-left. XXX means it isn't accessible
4 actions 0 (up), 1 (right), 2 (down), 3 (left)
probability of success is 0.8, going wrong (two sides) is 0.1
"""
trans = [[0, 0, 0, 0.9], [0, 0, 1, 0.1], [0, 1, 0, 0.2], [0, 1, 1, 0.8],
         [0, 2, 0, 0.9], [0, 2, 1, 0.1], [0, 3, 0, 1],
         [1, 0, 0, 0.1], [1, 0, 1, 0.8], [1, 0, 2, 0.1], [1, 1, 1, 0.1], [1, 1, 2, 0.8], [1, 1, 3, 0.1],
         [1, 2, 0, 0.1], [1, 2, 2, 0.1], [1, 2, 3, 0.8], [1, 3, 1, 0.1], [1, 3, 0, 0.8], [1, 3, 3, 0.1],
         [2, 0, 1, 0.1], [2, 0, 2, 0.9], [2, 1, 2, 0.9], [2, 1, 4, 0.1],
         [2, 2, 1, 0.1], [2, 2, 4, 0.8], [2, 2, 2, 0.1], [2, 3, 1, 0.8], [2, 3, 2, 0.1], [2, 3, 4, 0.1],
         [3, 0, 3, 0.1], [3, 0, 1, 0.8], [3, 0, 4, 0.1], [3, 1, 1, 0.1], [3, 1, 4, 0.8], [3, 1, 3, 0.1],
         [3, 2, 3, 0.9], [3, 2, 4, 0.1], [3, 3, 3, 0.9], [3, 3, 1, 0.1],

         [4, 0, 2, 0.8], [4, 0, 3, 0.1], [4, 0, 4, 0.1], [4, 1, 2, 0.1], [4, 1, 4, 0.9],
         [4, 2, 3, 0.1], [4, 2, 4, 0.9], [4, 3, 2, 0.1], [4, 3, 3, 0.8], [4, 3, 4, 0.1]
         ]

m = MDP(5, 4, [-0.04, -0.04, 1, -0.04, -1], trans, terminals=[False, False, True, False, True], gamma=0.5)

print(m.transitions)

m.value_iteration(utilities=[0, 0, 1, 0, -1], verbose=True)

m.policy_iteration(verbose=True)
