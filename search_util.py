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
    # data_file = "data/CS205_large_testdata__6.txt"
    
    data_array = np.loadtxt(data_file)
    feature_list = [2, 10, 9]
    # feature_list = [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 19, 21, 22, 23, 24, 25, 26, 27, 28, 31, 34, 36, 38, 40, 41, 45, 46, 47, 50, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 66, 67, 69, 70, 72, 73, 74, 76, 78, 80, 81, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 100, 101, 103, 105, 106, 107, 109, 110, 111, 114, 115, 116, 117, 118, 119, 120, 123, 124, 127, 128, 129, 130, 131, 133, 134, 135, 136, 140, 141, 144, 145, 146, 147, 148, 152, 153, 154, 155, 156, 157, 158, 163, 164, 165, 167, 168, 171, 173, 174, 176, 177, 178, 179, 180, 181, 182, 185, 186, 191, 192, 193, 194, 195, 196, 197, 198, 200, 203, 204, 205, 206, 207, 208, 211, 212, 213, 215, 216, 219, 220, 221, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 249]
    # feature_list = [2]
    acc = cross_validation_vectorize(data_array,feature_list)
    print(acc)
    # cross_validation_naive(data_array,feature_list)
