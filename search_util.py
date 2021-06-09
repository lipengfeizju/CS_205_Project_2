import numpy as np
from numpy import linalg as LA

def cross_validation_naive(data,feature_list):

    accuracy = 0
    label    = data[:,0].astype(int)
    feature  = data[:,feature_list]

    # print(feature.shape)
    num_pts = feature.shape[0]

    correct = 0
    for i in range(num_pts):
        
        feat_i = feature[i,:]
        min_dis = np.inf
        min_ind = -1
        for j in range(num_pts):
            if i == j: continue
            dis_vec = feature[j,:] - feat_i
            # this is the matrix multiplication between two vectors 
            # equavelant to dis_vec*dis_vec' in Matlab
            dis = np.sqrt(dis_vec @ dis_vec)
            if dis < min_dis: 
                min_dis = dis
                min_ind = j
        label_i = label[i]
        label_j = label[min_ind]
        correct +=  int(label_i==label_j)
    
    accuracy = correct/num_pts
    # print(accuracy)

    return accuracy

def cross_validation_vectorize(data,feature_list):
    accuracy = 0
    label    = data[:,0].astype(int)
    feature  = data[:,feature_list]

    # print(feature.shape)
    num_pts = feature.shape[0]

    correct = 0
    for i in range(num_pts):
        
        feat_i = feature[i,:]
        label_i = label[i]

        dist_vec = feature - feat_i
        dist_scale = LA.norm(dist_vec, axis=1)
        dist_scale[i] = np.inf
        min_ind = np.argmin(dist_scale)
        # print(dist_scale[min_ind])
        label_j = label[min_ind]
        correct +=  int(label_i==label_j)
    
    accuracy = correct/num_pts
    # print(accuracy)

    return accuracy

if __name__ == "__main__":

    data_file = "data/CS205_small_testdata__10.txt"
    data_array = np.loadtxt(data_file)
    feature_list = [2, 10, 9]
    # feature_list = [2]
    cross_validation_vectorize(data_array,feature_list)
    # cross_validation_naive(data_array,feature_list)
