from collections import deque
import random

class ReplayMemory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def add(self, transition):
        self.memory.append(transition)

    def get_random_sample(self, size):
        return random.sample(self.memory, size)

    def get_current_size(self):
        return len(self.memory)