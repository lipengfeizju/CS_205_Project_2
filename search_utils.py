import numpy as np
from nearest_neighbor import cross_validation_vectorize as cross_validation

def format_str(feature_list):
    set_str = "{"
    if len(feature_list) == 1:
        set_str += "{:d}".format(feature_list[0])
    elif  len(feature_list) > 1:
        for fi in feature_list: 
            set_str += "{:d}, ".format(fi)
        set_str = set_str[:-2]
    
    set_str += "}"
    return set_str

def load_data(data_path):
    data_array = np.loadtxt(data_path)

    feat_array = data_array[:,1:]
    # Normalization
    row_mean = np.mean(feat_array, axis = 0)
    feat_array = feat_array - row_mean.reshape([1,-1])
    # row_std = np.std(feat_array, axis=1)
    # feat_array = feat_array/row_std.reshape([-1,1])

    data_array[:,1:] = feat_array
    return data_array

def forward_search(data_array, truncate_level=0):

    num_pts, num_feat = data_array.shape
    num_feat -= 1 # the first column is the label, so number of feature -1
    print("This dataset has {:d} features (not including the class attribute), with {:d} instances.".format(num_feat, num_pts))

    all_acc = cross_validation(data_array,[i+1 for i in range(num_feat)])
    print("Running nearest neighbor with all {:d} features, using “leaving-one-out” evaluation, I get an accuracy of {:.2f}%".format(num_feat, all_acc*100))

    feature_list = []
    print("Beginning search.")
    best_acc_overall = 0
    best_feature_list = []
    if truncate_level == 0: truncate_level = num_feat
    for i in range(1,truncate_level+1):
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
    print("\nFinished search!! The best feature subset is "+ format_str(best_feature_list) + ", which has an accuracy of {:.2f}%".format(100*best_acc_overall))

def backward_search(data_array, truncate_level=0):

    num_pts, num_feat = data_array.shape
    num_feat -= 1 # the first column is the label, so number of feature -1
    print("This dataset has {:d} features (not including the class attribute), with {:d} instances.".format(num_feat, num_pts))

    all_acc = cross_validation(data_array,[i+1 for i in range(num_feat)])
    print("Running nearest neighbor with all {:d} features, using “leaving-one-out” evaluation, I get an accuracy of {:.2f}%".format(num_feat, all_acc*100))

    feature_list = [j for j in range(1,num_feat+1)]
    print("Beginning search.")
    best_acc_overall = 0
    best_feature_list = feature_list.copy()
    
    if truncate_level == 0: truncate_level = num_feat
    for i in range(1,truncate_level):
        best_acc_level = 0
        best_j = -1
        for j in range(1,num_feat+1):
            if j not in feature_list: continue
            feature_list_new = feature_list.copy()
            feature_list_new.remove(j)
            acc_j = cross_validation(data_array,feature_list_new)
            set_str = format_str(feature_list_new)
            # print("        Using feature(s) "+ set_str +" accuracy is {:.2f}%".format(100*acc_j))
            if acc_j > best_acc_level:
                best_j = j 
                best_acc_level = acc_j
    
        feature_list.remove(best_j)
        print("Feature set "+format_str(feature_list)+" was best, accuracy is {:.2f}%".format(100*best_acc_level))
        if best_acc_level > best_acc_overall:
            best_acc_overall  = best_acc_level
            best_feature_list = feature_list.copy()
    # 
    print("\nFinished search!! The best feature subset is "+ format_str(best_feature_list) + ", which has an accuracy of {:.2f}%".format(100*best_acc_overall))
