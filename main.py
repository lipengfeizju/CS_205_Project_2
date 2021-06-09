import numpy as np
from search_util import cross_validation_naive as cross_validation
import time


def format_str(feature_list):
    set_str = "{"
    if len(feature_list) == 1:
        set_str += "{:d}".format(feature_list[0])
    else: 
        for fi in feature_list: 
            set_str += "{:d}, ".format(fi)
        set_str = set_str[:-2]
    
    set_str += "}"
    return set_str

def forward_search(data_array):

    num_pts, num_feat = data_array.shape
    num_feat -= 1 # the first column is the label, so number of feature -1
    print("This dataset has {:d} features (not including the class attribute), with {:d} instances.".format(num_feat, num_pts))

    feature_list = []
    print("Beginning search.")
    best_acc_overall = 0
    best_feature_list = []
    
    for i in range(1,num_feat+1):
        best_acc_level = 0
        best_j = -1
        for j in range(1,num_feat+1):
            if j in feature_list: continue
            feature_list_new = feature_list + [j]
            acc_j = cross_validation(data_array,feature_list_new)
            set_str = format_str(feature_list_new)
            print("        Using feature(s) "+ set_str +" accuracy is {:.2f}%".format(100*acc_j))
            if acc_j > best_acc_level:
                best_j = j 
                best_acc_level = acc_j
    
        feature_list += [best_j]
        print("Feature set "+format_str(feature_list)+" was best, accuracy is {:.2f}%".format(100*best_acc_level))
        if best_acc_level > best_acc_overall:
            best_acc_overall  = best_acc_level
            best_feature_list = feature_list.copy()
    # 
    print("Finished search!! The best feature subset is "+ format_str(best_feature_list) + ", which has an accuracy of {:.2f}".format(best_acc_overall))

def backward_search(data_array):

    num_pts, num_feat = data_array.shape
    num_feat -= 1 # the first column is the label, so number of feature -1
    print("This dataset has {:d} features (not including the class attribute), with {:d} instances.".format(num_feat, num_pts))

    feature_list = [j for j in range(1,num_feat+1)]
    print("Beginning search.")
    best_acc_overall = 0
    best_feature_list = feature_list.copy()
    
    for i in range(1,num_feat+1):
        best_acc_level = 0
        best_j = -1
        for j in range(1,num_feat+1):
            if j not in feature_list: continue
            feature_list_new = feature_list.copy()
            feature_list_new.remove(j)
            acc_j = cross_validation(data_array,feature_list_new)
            set_str = format_str(feature_list_new)
            print("        Using feature(s) "+ set_str +" accuracy is {:.2f}%".format(100*acc_j))
            if acc_j > best_acc_level:
                best_j = j 
                best_acc_level = acc_j
    
        feature_list.remove(best_j)
        print("Feature set "+format_str(feature_list)+" was best, accuracy is {:.2f}%".format(100*best_acc_level))
        if best_acc_level > best_acc_overall:
            best_acc_overall  = best_acc_level
            best_feature_list = feature_list.copy()
    # 
    print("Finished search!! The best feature subset is "+ format_str(best_feature_list) + ", which has an accuracy of {:.2f}".format(best_acc_overall))


if __name__ == "__main__":
    # data_file = "data/CS205_small_testdata__28.txt"
    data_file = "data/CS205_large_testdata__1.txt"
    # feature_list = [2, 10, 9]
    
    
    data_array = np.loadtxt(data_file)

    start_time = time.time()
    forward_search(data_array)
    print("--- %s seconds ---" % (time.time() - start_time))
    # backward_search(data_array)

    
    