import numpy as np
import time
from search_utils import forward_search,backward_search

if __name__ == "__main__":
    data_file = "data/CS205_small_testdata__28.txt"
    # data_file = "data/CS205_large_testdata__6.txt"
    # feature_list = [2, 10, 9]
    
    # data_array = np.loadtxt(data_file)

    # start_time = time.time()
    # forward_search(data_array)
    # # backward_search(data_array)
    # print("--- %s seconds ---" % (time.time() - start_time))
    print("Welcome to My Feature Selection Algorithm.")
    file_name = input("Type in the name of the file to test (e.g. CS205_small_testdata__28.txt):")
    data_file = "data/" + file_name
    data_array = np.loadtxt(data_file)
    num_feat = data_array.shape[1]-1
    truncate_level = 0
    if num_feat > 5:
        print("You have {:d} features (not including the class attribute), do you want early stop?".format(num_feat))
        t_level = input("If you want, please type in a positive integar, otherwise, just press enter and continue:")
        if t_level != '': 
            t_level = int(t_level)
            if t_level > 0: truncate_level = t_level

    print("Type the number of the algorithm you want to run. \n    1) Forward Selection\n    2) Backward Elimination")
    x = input()
    selected_method = int(x)
    if selected_method == 1:
        forward_search(data_array)
    elif selected_method ==2 :
        backward_search(data_array)
    else:
        print("Please input a valid method! (1 or 2)")






    
    
