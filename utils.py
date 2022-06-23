import os
import sys
import json
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


from params import *
import MNIST

if sys.version_info >= (3,):
    xrange = range

if is_gpu is True:
    mat = cp
else:
    mat = np

weight_cmap = 'bone' 
 
def sigma(x):
    '''
    sigmoid function
    '''
    return 1/(1 + mat.exp(-x))

def deriv_sigma(x):
    '''
    derivative of sigmoid function
    '''
    return sigma(x)*(1.0 - sigma(x))

def shuffle_arrays(*args):
    permutation = mat.random.permutation(args[0].shape[1])
    res = (arg[:, permutation] for arg in args)
    return res

def plot_weights(weights, save_dir=None, suffix=None, normalize=False):
    '''
    weights : List of weight matrices to plot.
    normalize:  Whether to normalize each receptive field. 
    '''

    def cal_prime_factor(n):
        res = []
        stopnum = n
        if n == 1: 
            return [1]
        while 1:
            if stopnum == 1:
                break
            temp = 2
            while 1:
                if stopnum % temp == 0:
                    break
                temp += 1
            res.append(temp)
            stopnum /= temp
        return res

    def find_closest_divisors(n):
        # Find divisors of a number n that are closest to its square root.
        a_max = mat.floor(mat.sqrt(n))
        if n % a_max == 0:
            a = a_max
            b = n/a
        else:
            factors = cal_prime_factor(n)
            candidates = mat.array([1])
            for i in xrange(len(factors)):
                f = factors[i]
                candidates = mat.union1d(candidates, f*candidates)
                candidates[candidates > a_max] = 0
            a = candidates.max()
            b = n/a
        return (int(a), int(b))

    plt.close('all')

    fig = plt.figure(figsize=(18, 9))

    num_weights = len(weights)
    n = [weight.shape[0] for weight in weights]
    n_in = weights[0].shape[-1]

    print(num_weights, n)

    grid_specs = [0]*num_weights
    axes = [ [0]*i for i in n ]

    max_Ws = [ mat.amax(weight) for weight in weights ]
    min_Ws = [ mat.amin(weight) for weight in weights ]

    W_sds = [ mat.std(weight) for weight in weights ]
    W_avgs = [ mat.mean(weight) for weight in weights ]

    for m in xrange(num_weights):
        print("Layer {0} | W_avg: {1:.6f}, W_sd: {2:.6f}.".format(m, mat.mean(weights[m]), mat.std(weights[m])))

    for m in xrange(num_weights):
        if m == 0:
            img_Bims = find_closest_divisors(n_in)
        else:
            img_Bims = grid_dims

        grid_dims = find_closest_divisors(n[m])
        grid_dims = (grid_dims[1], grid_dims[0])
        grid_specs[m] = gs.GridSpec(grid_dims[0], grid_dims[1])

        for k in xrange(n[m]):
            row = k // grid_dims[1]
            col = k - row*grid_dims[1]

            axes[m][k] = fig.add_subplot(grid_specs[m][row, col])
            if normalize:
                heatmap = axes[m][k].imshow(weights[m][k].reshape(img_Bims).T, interpolation="nearest", cmap='bone')
            else:
                heatmap = axes[m][k].imshow(weights[m][k].reshape(img_Bims).T, interpolation="nearest", vmin=W_avgs[m] - 3.465*W_sds[m], vmax=W_avgs[m] + 3.465*W_sds[m], cmap='bone')
            axes[m][k].set_xticklabels([])
            axes[m][k].set_yticklabels([])
            axes[m][k].tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off') 

            if m == num_weights-1 and k == 0:
                plt.colorbar(heatmap)

        grid_specs[m].update(left=float(m)/num_weights, right=(m+1.0)/num_weights, hspace=1.0/(grid_dims[0]), wspace=0.05, bottom=0.02, top=0.98)

    if save_dir != None:
        if suffix != None:
            plt.savefig(save_dir + 'weights' + suffix + '.png')
        else:
            plt.savefig(save_dir + 'weights.png')
    else:
        plt.show()

def load_MNIST(n_valid=0):
    try:
        train_x = mat.load("train_x.npy")
        test_x  = mat.load("test_x.npy")
        train_y = mat.load("train_y.npy")
        test_y  = mat.load("test_y.npy")
        if n_valid != 0:
            valid_x = mat.load("valid_x.npy")
            valid_y = mat.load("valid_y.npy")
    except:
        print("MNIST .npy files not found.\nLooking for original files...")
        try:
            path = 'MNIST'
            #train_set = NMNIST(path, train=True, data_type='event')
            #print("load data to mat")
            #NMNIST.read_bin_save_to_np('train-images.idx3-ubyte','train_x.npy')
            #NMNIST.read_bin_save_to_np('t10k-images.idx3-ubyte','test_x.npy')
            #NMNIST.read_bin_save_to_np('train-labels.idx1-ubyte','train_y.npy')
            #NMNIST.read_bin_save_to_np('t10k-labels.idx1-ubyte','test_y.npy')
            
            if n_valid != 0:
                train_x, train_y, test_x, test_y, valid_x, valid_y = get_MNIST(n_valid)
                save_MNIST(train_x, train_y, test_x, test_y, valid_x, valid_y)
            else:
                train_x, train_y, test_x, test_y = get_MNIST()
                save_MNIST(train_x, train_y, test_x, test_y)
            
        except:
            return

        print(".npy files saved.\n")

    if n_valid != 0:
        return train_x, train_y, test_x, test_y, valid_x, valid_y
    else:
        return train_x, train_y, test_x, test_y

def save_MNIST(train_x, train_y, test_x, test_y, valid_x=None, valid_y=None):

    mat.save("train_x.npy", train_x)
    mat.save("test_x.npy", test_x)
    mat.save("train_y.npy", train_y)
    mat.save("test_y.npy", test_y)

    if valid_x != None and valid_y != None:
        mat.save("valid_x.npy", valid_x)
        mat.save("valid_y.npy", valid_y)

def get_MNIST(n_valid=0):
    '''
    Original MNIST binary files (http://yann.lecun.com/exdb/mnist/)
    '''

    if (os.path.isfile("train-images.idx3-ubyte") and
        os.path.isfile("train-labels.idx1-ubyte") and
        os.path.isfile("t10k-images.idx3-ubyte") and
        os.path.isfile("t10k-labels.idx1-ubyte")):
        print("Found original MNIST files. Converting to .npy files...")
        try:
            print("MNIST")
            trainfeatures, trainlabels = MNIST.traindata()
            testfeatures, testlabels   = MNIST.testdata()
        except:
            print("Error: Could not convert original MNIST files.")
            return
    else:
        print("Error: Could not find original MNIST files.")
        return
 
    # normalizatoin
    N_CONST = 255.0
    if n_valid > 0:
        valid_x = trainfeatures[:, :n_valid]/N_CONST
         
    train_x = trainfeatures[:, n_valid:]/N_CONST
    test_x   = testfeatures/N_CONST
 
    n_train = trainlabels.size - n_valid
 
    # generate vectors of dataset
    V_CONST = 1
    NUM_CATEGORIES = 10
    if n_valid > 0:
        valid_y = mat.zeros((NUM_CATEGORIES, n_valid))
        for i in range(n_valid):
            valid_y[int(trainlabels[i]), i] = V_CONST
 
    train_y = mat.zeros((NUM_CATEGORIES, n_train))
    for i in xrange(n_train):
        train_y[int(trainlabels[n_valid + i]), i] = V_CONST
 
    n_test = testlabels.size
    test_y = mat.zeros((NUM_CATEGORIES, n_test))
    for i in xrange(n_test):
        test_y[int(testlabels[i]), i] = V_CONST
 
    if n_valid <= 0:
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y, test_x, test_y, valid_x, valid_y
