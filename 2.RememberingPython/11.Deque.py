print("# Deque ------------------------------------")

"""
We determine the size at the begginig

"""

from collections import deque

dq = deque(maxlen=10)

# Add elemet to end
dq.append(1)
dq.append(2)
dq.append(3)
print(dq)

# Add element at the beggining
dq.appendleft(0)
print(dq)
