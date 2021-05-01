# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla
#Modified: Alex Porter
import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return distance.cityblock(u, v)

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset in which each row is an image patch.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 3):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

def lsh_search1(A, hashed_A, functions, query_index, num_neighbors = 3):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[1] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    distances = []
    k_neigh = []
    query_vec = A[query_index]
    for vec in A:
        if np.array_equal(vec, query_vec):
            pass
        else:
            dist = l1(vec, query_vec)
            distances.append((vec, dist))
    sorted_dist = sorted(distances, key = lambda x : x[1])[:num_neighbors]
    return [t[0] for t in sorted_dist]

def linear_search1(A, query_index, num_neighbors):
    distances = []
    k_neigh = []
    query_vec = A[query_index]
    for vec in A:
        if np.array_equal(vec, query_vec):
            pass
        else:
            dist = l1(vec, query_vec)
            distances.append((vec, dist))
    sorted_dist = sorted(distances, key = lambda x : x[1])[:num_neighbors]
    return [t[1] for t in sorted_dist]

def linear_search2(A, query_index, num_neighbors):
    distances = []
    k_neigh = []
    query_vec = A[query_index]
    for i in range(len(A)):
        vec = A[i]
        if np.array_equal(vec, query_vec):
            pass
        else:
            dist = l1(vec, query_vec)
            distances.append((i, dist))
    sorted_dist = sorted(distances, key = lambda x : x[1])[:num_neighbors]
    return [t[0] for t in sorted_dist]

def error(A, f, hashed, i = 3, j = 10):
    error = 0
    for idj in range(1, j + 1):
        idx = idj*100 - 1
        lsh = lsh_search1(A, hashed, f, idx, i)
        dist_lsh = np.sum(lsh)
        lin = linear_search1(A, idx, i)
        dist_lin = np.sum(lin)
        error = error + (dist_lsh / dist_lin)
    return error / 10


#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


#####################################################################################################
#####################################################################################################
#####################################################################################################

def main(problem):
    A = load_data("data/patches.csv")
    ##### AVERAGE TIME #####
    if problem == "time":
        LSH_time = []
        lin_time = []
        f, hashed = lsh_setup(A)
        for i in range(99, 1000, 100):
            start_time = time.time()
            lsh_search(A, hashed, f, i, 3)
            middle_time = time.time()
            LSH_time.append(middle_time - start_time)
            linear_search(A, i, 3)
            end_time = time.time()
            lin_time.append(end_time - middle_time)

        print("-------------------------------------------------------")
        print("Average search time for LSH is " + str(np.average(LSH_time)) + " seconds")
        print("Average search time for linear search is " + str(np.average(lin_time)) + " seconds")
        print("-------------------------------------------------------")
    
        top10_lsh, top10_lin = None

    ##### ERROR RATE #####
    if problem == "error":
        error_L = []
        for L in range(10, 21, 2):
            f, hashed = lsh_setup(A, 24, L)
            err = error(A, f, hashed, 3, 10)
            error_L.append(err)
        print(error_L)

        error_k = []
        for k in range(16, 25, 2):
            f, hashed = lsh_setup(A, k, 10)
            err = error(A, f, hashed, 3, 10)
            error_k.append(err)

        print(error_k)

        top10_lsh, top10_lin = None

    ##### TOP 10 NEIGH #####
    if problem == "top10":
        f, hashed = lsh_setup(A)
        top10_lsh = lsh_search(A, hashed, f, 99, 10)
        top10_lin = linear_search2(A, 99, 10)

    return top10_lsh, top10_lin
        


################### RESULTS ###################

# PROBLEM 1: 

#main("time")
"""
-------------------------------------------------------
Average search time for LSH is 0.1394000768661499 seconds
Average search time for linear search is 0.9721627235412598 seconds
-------------------------------------------------------
"""

# PROBLEM 2:
#main("error")

error_L = [1.1100124060238752, 1.0765273347531996, 1.0704966476143745, 1.0608022526489822, 1.0210564593929072, 0.9258718168296245]
error_k = [1.0189710776022127, 1.0487689994040577, 1.0766954236694721, 1.0674361420591052, 1.0924846800897563]

"""
# Plot problem 2:

plt.figure(figsize=(15, 7))
plt.plot([10, 12, 14, 16, 18, 20], error_L)
plt.ylabel('Error')
plt.xlabel('L')
plt.title("Error as function of L, k = 24")
plt.savefig("L.png")
plt.show()

plt.figure(figsize=(15, 7))
plt.plot([16, 18, 20, 22, 24], error_k)
plt.ylabel('Error')
plt.xlabel('k')
plt.title("Error as function of k, L = 10")
plt.savefig("k.png")
plt.show()
"""


# PROBLEM 3:
#lsh_10, lin_10 = main("top10")

A = load_data("data/patches.csv")

lsh_10 = [23633, 26168, 23843, 17800, 25888, 28104, 5914, 13038, 12211, 35732]
lin_10 = [58690, 23633, 26168, 38169, 24692, 48596, 37742, 37252, 28054, 15852]

"""
# Plot problem 3:

plot(A, lsh_10, "lsh")
plot(A, lin_10, "lin")

plot(A, [99], "lsh_orig")
"""