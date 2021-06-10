import numpy as np
import time
from search_utils import forward_search,backward_search

if __name__ == "__main__":
    data_file = "data/CS205_small_testdata__28.txt"
    data_array = np.loadtxt(data_file)

    start_time = time.time()
    print("-"*20 + "Forward Search" + "-"*20)
    forward_search(data_array)
    print("----- %f seconds -----\n" % (time.time() - start_time))

    start_time = time.time()
    print("-"*20 + "Backward Search" + "-"*20)
    backward_search(data_array)
    print("----- %f seconds -----" % (time.time() - start_time))