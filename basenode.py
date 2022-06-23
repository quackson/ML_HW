import numpy as np
import cupy as cp
from abc import abstractmethod
from spikingjelly.clock_driven import encoding
from spikingjelly.datasets.n_mnist import NMNIST
import torch
from params import *

if is_gpu is True:
    mat = cp
else:
    mat = np

class Guerguiev2017BaseNode:
    def __init__(self, model_net, m, inputS):
        
        LOAD_CONST=1  #const number for initial load
        self.net = model_net 
        self.m = m #list(hidden nodes, ... , final nodes)
        self.size = self.net.n[m]
        self.f_input_size = inputS        
        self.v_basal  = mat.zeros((self.size, LOAD_CONST))
        self.v_soma = mat.zeros((self.size, LOAD_CONST))
        self.spiking_rate = mat.zeros((self.size, LOAD_CONST))
        self.spike_hist = mat.zeros((self.size, mem), dtype=mat.int8)
        self.E       = mat.zeros((self.size, LOAD_CONST))
        self.delta_W = mat.zeros(self.net.W[self.m].shape)
        self.delta_b = mat.zeros((self.size, LOAD_CONST))
        self.average_v_soma_f        = mat.zeros((self.size, LOAD_CONST))
        self.average_v_soma_t        = mat.zeros((self.size, LOAD_CONST))
        self.average_spiking_rate_f = mat.zeros((self.size, LOAD_CONST))
        self.average_spiking_rate_t = mat.zeros((self.size, LOAD_CONST))
        self.average_basal_input_f    = mat.zeros((self.f_input_size, LOAD_CONST))
        self.integration_counter = 0
        self.create_integration_vars()

    def origin(self):
        
        CLEAR_CONST=0
        self.v_basal  *= CLEAR_CONST
        self.v_soma   *= CLEAR_CONST
        self.spiking_rate  *= CLEAR_CONST
        self.spike_hist    *= CLEAR_CONST
        self.basal_input_hist   *= CLEAR_CONST
        self.v_soma_hist        *= CLEAR_CONST
        self.spiking_rate_hist  *= CLEAR_CONST
        self.E   *= CLEAR_CONST
        self.delta_W *= CLEAR_CONST
        self.delta_b *= CLEAR_CONST
        self.average_v_soma_f  *= CLEAR_CONST
        self.average_v_soma_t  *= CLEAR_CONST
        self.average_spiking_rate_f *= CLEAR_CONST
        self.average_spiking_rate_t *= CLEAR_CONST
        self.average_basal_input_f  *= CLEAR_CONST
        self.integration_counter = CLEAR_CONST

    def create_integration_vars(self):
        self.basal_input_hist    = mat.zeros((self.f_input_size, integration_time))
        self.v_soma_hist        = mat.zeros((self.size, integration_time))
        self.spiking_rate_hist = mat.zeros((self.size, integration_time))

    
   
    @abstractmethod
    def update_W(self):
        """
        define loss functions, calculate gradients of loss, and update weights
        """
        raise NotImplementedError
    
    
    @abstractmethod
    def update_v_soma(self, phase):
        '''
        Update somatic potentials & calculate firing rates.
        '''
        raise NotImplementedError
    
    @abstractmethod
    def out_f(self, f_input, b_input):
        '''
        Perform a forward phase
        '''
        raise NotImplementedError
    
    @abstractmethod
    def out_t(self, f_input, b_input):
        '''
        Perform a target phase
        '''
        raise NotImplementedError

    def update_v_basal(self, f_input):

        if not use_spiking_feedforward:
            self.basal_input = f_input
        else:
            self.basal_input = mat.dot(f_input, kappas)
        

        pos=self.integration_counter % integration_time
        hist_temp=self.basal_input[:, 0]
        self.basal_input_hist[:, pos] = hist_temp

        self.v_basal = self.net.W[self.m] @ self.basal_input + self.net.b[self.m]
    
    def calc_averages(self, phase):
        if phase == "forward":
            self.average_v_soma_f        = mat.mean(self.v_soma_hist, axis=-1)[:, mat.newaxis]
            self.average_spiking_rate_f = mat.mean(self.spiking_rate_hist, axis=-1)[:, mat.newaxis]
            self.average_basal_input_f    = mat.mean(self.basal_input_hist, axis=-1)[:, mat.newaxis]
        elif phase == "target":
            self.average_v_soma_t        = mat.mean(self.v_soma_hist, axis=-1)[:, mat.newaxis]
            self.average_spiking_rate_t = mat.mean(self.spiking_rate_hist, axis=-1)[:, mat.newaxis]

    def update_step(self):
        self.integration_counter += 1
        self.integration_counter %= integration_time

    def spike(self):
        '''
        spiking Jelly encoding
        '''
        pe = encoding.PoissonEncoder()
    
        x=self.spiking_rate
        out_spike = pe(torch.tensor(x))
        out_spike = mat.array(out_spike)
        #print(out_spike.size)
        self.spike_hist = mat.concatenate([self.spike_hist[:, 1:], out_spike], axis=-1)