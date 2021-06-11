import time
from search_utils import forward_search,backward_search, load_data

if __name__ == "__main__":
    # data_file = "data/CS205_large_testdata__6.txt"
    data_file = "data/CS205_small_testdata__33.txt"

    data_array = load_data(data_file)

    start_time = time.time()
    print("-"*20 + "Forward Search" + "-"*20)
    forward_search(data_array, 10)
    print("----- %f seconds -----\n" % (time.time() - start_time))

    start_time = time.time()
    print("-"*20 + "Backward Search" + "-"*20)
    backward_search(data_array,10)
    print("----- %f seconds -----" % (time.time() - start_time))