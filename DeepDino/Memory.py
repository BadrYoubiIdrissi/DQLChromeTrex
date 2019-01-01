from collections import deque
import numpy as np

"""
Memory class. Takes a max_size and let's us add items one 
by one until reaching the max_size and then removes the oldest items.
We do this with a python dequeu which has O(1) cost for removing items from queue. 
"""
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        self.size = max_size
    
    def add(self, experience):
        #Automatically removes oldest items when full
        self.buffer.append(experience) 
    
    """
    Returns a random sample of size batch_size from the memory  
    """
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]