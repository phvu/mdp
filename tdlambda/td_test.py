from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import print_function

from tdlambda.td import TDLambda, Move


solver = TDLambda(state_count=6)

episodes = [[Move(1, 1), Move(3, 0), Move(4, 1), Move(0, 0)],
            [Move(1, 1), Move(3, 0), Move(5, 10), Move(0, 0)],
            [Move(1, 1), Move(3, 0), Move(4, 1), Move(0, 0)],
            [Move(1, 1), Move(3, 0), Move(4, 1), Move(0, 0)],
            [Move(2, 2), Move(3, 0), Move(5, 10), Move(0, 0)]]
print(solver.run(episodes, gamma=1, learning_rate=lambda t: 1))
print(solver.run(episodes, gamma=1, lambda_val=1))
print(solver.run(episodes, gamma=1, lambda_val=0))
print(solver.run(episodes, gamma=1, lambda_val=0.5))
print(solver.run(episodes, gamma=0.9, lambda_val=0.8))
