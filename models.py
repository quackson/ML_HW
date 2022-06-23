import os
import sys
import time
import json
import shutil
import datetime
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from spikingjelly.clock_driven import encoding
from spikingjelly.datasets.n_mnist import NMNIST
import torch

from basenode import Guerguiev2017BaseNode
from params import *
from utils import load_MNIST, shuffle_arrays, sigma, deriv_sigma

if sys.version_info >= (3,):
    xrange = range

if is_gpu is True:
    mat = cp
else:
    mat = np

class Guerguiev2017hiddenNode(Guerguiev2017BaseNode):
    def __init__(self, model_net, m, input_S, b_input_S):
        '''
        Initialize a hidden layer for the model
        '''

        super().__init__(model_net, m, input_S)

        LOAD_CONST=1

        self.b_input_size = b_input_S
        self.v_apical  = mat.zeros((self.size, LOAD_CONST))
        self.delta_Y = mat.zeros(self.net.Y[self.m].shape)
        self.average_v_apical_f = mat.zeros((self.size, LOAD_CONST))
        self.average_v_apical_t = mat.zeros((self.size, LOAD_CONST))
        if update_feedback_weights:
            self.average_apical_input_f = mat.zeros((self.b_input_size, LOAD_CONST))        
        self.alpha_f  = mat.zeros((self.size, LOAD_CONST))
        self.alpha_t  = mat.zeros((self.size, LOAD_CONST))
        self.v_apical_hist  = mat.zeros((self.size, integration_time))
        self.apical_input_hist = mat.zeros((self.b_input_size, integration_time))

    def origin(self):

        super().origin() #load origin base layel
        CLEAR_CONST=0
        self.v_apical  *= CLEAR_CONST
        self.v_apical_hist   *= CLEAR_CONST
        self.apical_input_hist  *= CLEAR_CONST
        self.delta_Y *= 0
        self.average_v_apical_f  *= CLEAR_CONST
        self.average_v_apical_t  *= CLEAR_CONST
        if update_feedback_weights:
            self.average_apical_input_f *= CLEAR_CONST
        self.alpha_f   *= CLEAR_CONST
        self.alpha_t   *= CLEAR_CONST

    def update_W(self):

        v_temp=k_B*lambda_max*deriv_sigma(self.average_v_soma_f)
        if not use_backprop:
            if record_backprop_angle and not use_backprop and calc_E_bp:
                self.E_bp = mat.dot(self.net.W[self.m+1].T, self.net.l[self.m+1].E_bp)*v_temp        
            self.E = (self.alpha_t - self.alpha_f)*-v_temp
        else:
            self.E_bp = mat.dot(self.net.W[self.m+1].T, self.net.l[self.m+1].E_bp)*v_temp
            self.E = self.E_bp

        if record_backprop_angle and (not use_backprop) and calc_E_bp:
            self.delta_b_bp = self.E_bp

        self.delta_W = mat.dot(self.E, self.average_basal_input_f.T)
        self.net.W[self.m] += -self.net.f_lr[self.m]*P_hidden*self.delta_W

        self.delta_b = self.E
        self.net.b[self.m] += -self.net.f_lr[self.m]*P_hidden*self.delta_b

    def update_Y(self):

        v_temp=deriv_sigma(self.average_v_apical_f)
        E_inv = (lambda_max*sigma(self.average_v_soma_f) - self.alpha_f)*-v_temp

        self.delta_Y = E_inv @ self.average_apical_input_f.T
        self.net.Y[self.m] += (-self.net.b_lr[self.m]*self.delta_Y)
        if use_feedback_bias:
            self.delta_c  = E_inv
            self.net.c[self.m] += -self.net.b_lr[self.m]*self.delta_c

    def update_v_apical(self, b_input):
        
        self.apical_input = b_input
        if use_spiking_feedback:
            self.apical_input = self.apical_input @ kappas

        pos=self.integration_counter % integration_time

        self.apical_input_hist[:, pos] = self.apical_input[:, 0]
        self.v_apical = (self.net.Y[self.m] @ self.apical_input)
        if use_feedback_bias:
            self.v_apical += self.net.c[self.m]
        self.v_apical_hist[:, pos] = self.v_apical[:, 0]

    def update_v_soma(self):
        
        if use_conductances:
            self.v_soma_dot = -g_l*self.v_soma + g_b*(self.v_basal-self.v_soma)        
            if use_apical_conductance:
                self.v_soma_dot += g_a*(self.v_apical - self.v_soma)
            self.v_soma += self.v_soma_dot*dt
        else:
            self.v_soma = k_B*self.v_basal

        pos=self.integration_counter % integration_time

        self.v_soma_hist[:, pos] = self.v_soma[:, 0]
        self.spiking_rate = lambda_max*sigma(self.v_soma)
        self.spiking_rate_hist[:, pos] = self.spiking_rate[:, 0]

    def out_f(self, f_input, b_input):
        '''
        compute a forward phase pass
        '''

        self.update_v_basal(f_input)
        self.update_v_apical(b_input)
        self.update_v_soma()
        self.spike()
        self.update_step()

    def out_t(self, f_input, b_input):
        '''
        compute a target phase pass
        '''

        self.update_v_basal(f_input)
        self.update_v_apical(b_input)
        self.update_v_soma()
        self.spike()
        self.update_step()

    def plateau_f(self, plateau_indices):
        '''
        compute apical plateau potentials for forward phase 
        '''
        temp=mat.mean(self.v_apical_hist[plateau_indices], axis=-1)
        self.average_v_apical_f[plateau_indices] = temp[:, mat.newaxis]
        self.alpha_f[plateau_indices] = sigma(self.average_v_apical_f[plateau_indices])

    def plateau_t(self, plateau_indices):
        '''
        compute apical plateau potentials for target phase 
        '''

        temp=mat.mean(self.v_apical_hist[plateau_indices], axis=-1)
        self.average_v_apical_t[plateau_indices] = temp[:, mat.newaxis]
        self.alpha_t[plateau_indices] = sigma(self.average_v_apical_t[plateau_indices])

    def calc_averages(self, phase):
        '''
        compute averages
        '''
        super().calc_averages(phase)

        temp=mat.mean(self.apical_input_hist, axis=-1)
        if update_feedback_weights:
            if phase == "forward":
                self.average_apical_input_f = temp[:, mat.newaxis]
            elif phase == "target":
                self.average_apical_input_t = temp[:, mat.newaxis]

class Guerguiev2017finalNode(Guerguiev2017BaseNode):
    def __init__(self, net_model, m, input_S):
        '''
        Initialize a final layer for model
        '''

        super().__init__(net_model, m, input_S)
        self.I = mat.zeros((self.size, 1))

    def origin(self):
        '''
        refine origin vars
        '''
        super().origin()
        self.I *= 0

    def update_W(self):
        '''
        Update feedforward weights
        '''
        temp_v=k_D*lambda_max*deriv_sigma(self.average_v_soma_f)
        self.E = (self.average_spiking_rate_t - lambda_max*sigma(self.average_v_soma_f))*-temp_v

        if use_backprop :
            self.E_bp = (self.average_spiking_rate_t - lambda_max*sigma(self.average_v_soma_f))*-temp_v
        elif  record_backprop_angle and calc_E_bp :
            self.E_bp = (self.average_spiking_rate_t - lambda_max*sigma(self.average_v_soma_f))*-temp_v

        self.delta_W = (self.E @ self.average_basal_input_f.T)
        self.delta_b = self.E

        self.net.W[self.m] += -self.net.f_lr[self.m]*P_final*self.delta_W
        self.net.b[self.m] += -self.net.f_lr[self.m]*P_final*self.delta_b

    def update_I(self, b_input=None):
        '''
        Update injected perisomatic currents
        '''

        if b_input is None:
            self.I *= 0
        else:
            g_E = b_input
            g_I = -g_E + 1
            if not use_conductances:
                self.k_I  = g_I/(g_l + g_d + g_E + g_I)
                self.k_D2 = g_d/(g_l + g_d + g_E + g_I)
                self.k_E  = g_E/(g_l + g_d + g_E + g_I)
            else:
                self.I = g_E*(E_E - self.v_soma) + g_I*(E_I - self.v_soma)
            
                

    def update_v_soma(self, phase):
        '''
        Update somatic potentials
        '''
        forwardIF=False
        targetIF=False
        if phase == "forward":
            forwardIF=True
        elif phase == "target":
            targetIF=True

        if use_conductances:
            temp=-g_l*self.v_soma + g_d*(self.v_basal - self.v_soma)
            if forwardIF:
                self.v_soma_dot = temp
            elif targetIF:
                self.v_soma_dot = temp + self.I
            self.v_soma += self.v_soma_dot*dt
        else:
            if forwardIF:
                self.v_soma = k_D*self.v_basal
            elif targetIF:
                self.v_soma = self.k_D2*self.v_basal + self.k_I*E_I + self.k_E*E_E

        pos=self.integration_counter % integration_time
        self.v_soma_hist[:, pos] = self.v_soma[:, 0]
        self.spiking_rate = lambda_max*sigma(self.v_soma)
        self.spiking_rate_hist[:, pos] = self.spiking_rate[:, 0]

    def out_f(self, f_input, b_input):
        '''
        compute a forward phase pass
        '''

        self.update_v_basal(f_input)
        self.update_I(b_input)
        self.update_v_soma(phase="forward")
        self.spike()
        self.update_step()

    def out_t(self, f_input, b_input):
        '''
        compute a target phase pass
        '''

        self.update_v_basal(f_input)
        self.update_I(b_input)
        self.update_v_soma(phase="target")
        self.spike()
        self.update_step()

class Guerguiev2017Network:
    def __init__(self, n):
        '''
        Initialize a network
        '''

        if type(n) == int:
            n = (n,)
        self.n = n           
        self.M = len(self.n) 
        self.n_neurons_per_category = int(self.n[-1]/10)
        self.x_train, self.t_train, self.x_test, self.t_test = load_MNIST()
        self.n_in  = self.x_train.shape[0] 
        self.n_out = self.n[-1]  
        self.x_hist = mat.zeros((self.n_in, mem))
        self.current_epoch = None
        print("Creating network with "+str(self.M)+" layers.")

        self.init_weights()
        self.init_layers()

    def init_weights(self):
        '''
        Initialize weights 
        '''
        temp = xrange(self.M-1, -1, -1)
        if use_weight_optimization:                       
            nu    = lambda_max*0.25
            V_avg = 3                 
            V_sd  = 3                  
            b_avg = 0.8                
            b_sd  = 0.001               
            V_sm  = V_sd**2 + V_avg**2 

        self.W = [0]*self.M
        self.b = [0]*self.M
        self.Y = [0]*(self.M-1)
        if use_feedback_bias:
            self.c = [0]*(self.M-1)

        if use_sparse_feedback:
            self.Y_dropout_indices = [0]*(self.M-1)

        for m in temp:
            if m != 0:
                N = self.n[m-1]
            else:
                N = self.n_in

            if use_weight_optimization:
                W_avg = (V_avg - b_avg)/(nu*N*V_avg)
                W_sm  = (V_sm + (nu**2)*(N - N**2)*(W_avg**2)*(V_avg**2) - 2*N*nu*b_avg*V_avg*W_avg - (b_avg**2))/(N*(nu**2)*V_sm)
                W_sd  = mat.sqrt(W_sm - W_avg**2)
            
                self.W[m] = W_avg + 3.465*W_sd*mat.random.uniform(-1, 1, size=(self.n[m], N))
                self.b[m] = b_avg + 3.465*b_sd*mat.random.uniform(-1, 1, size=(self.n[m], 1))
            else:
                self.W[m] = 0.1*mat.random.uniform(-1, 1, size=(self.n[m], N))
                self.b[m] = 1.0*mat.random.uniform(-1, 1, size=(self.n[m], 1))

            if m != 0:
                if use_broadcast:
                    if use_weight_optimization:
                        self.Y[m-1] = W_avg + 3.465*W_sd*mat.random.uniform(-1, 1, size=(N, self.n[-1]))
                        if use_feedback_bias:
                            self.c[m-1] = b_avg + 3.465*b_sd*mat.random.uniform(-1, 1, size=(N, 1))
                    else:
                        self.Y[m-1] = mat.random.uniform(-1, 1, size=(N, self.n[-1]))
                        if use_feedback_bias:
                            self.c[m-1] = mat.random.uniform(-1, 1, size=(N, 1))
                else:
                    if use_weight_optimization:
                         self.Y[m-1] = W_avg + 3.465*W_sd*mat.random.uniform(-1, 1, size=(N, self.n[m]))
                         if use_feedback_bias:
                             self.c[m-1] = b_avg + 3.465*b_sd*mat.random.uniform(-1, 1, size=(N, 1)) 
                    else:
                        self.Y[m-1] = mat.random.uniform(-1, 1, size=(N, self.n[m]))
                        if use_feedback_bias:
                            self.c[m-1] = mat.random.uniform(-1, 1, size=(N, 1))

        if use_symmetric_weights == True:
            self.make_weights_symmetric()

        if use_sparse_feedback:
            for m in xrange(self.M-1):
                self.Y_dropout_indices[m] = mat.random.choice(len(self.Y[m].ravel()), int(0.8*len(self.Y[m].ravel())), False)
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0
                self.Y[m] *= 5

        self.print_weights()

    def print_weights(self):
        print("Current network weights:")
        for m in xrange(self.M-1, -1, -1):
            print("Layer {0} -- {1} units.".format(m, self.n[m]))
            print("\tW_avg: {0:.6f},\tW_sd: {1:.6f}.".format(mat.mean(self.W[m]), mat.std(self.W[m])))
            print("\tb_avg: {0:.6f},\tb_sd: {1:.6f}.".format(mat.mean(self.b[m]), mat.std(self.b[m])))
            if m != self.M-1:
                print("\tY_avg: {0:.6f},\tY_sd: {1:.6f}.".format(mat.mean(self.Y[m]), mat.std(self.Y[m])))
                if use_feedback_bias:
                    print("\tc_avg: {0:.6f},\tc_sd: {1:.6f}.".format(mat.mean(self.c[m]), mat.std(self.c[m])))

    def make_weights_symmetric(self):

        temp = xrange(self.M-2, -1, -1)

        if use_broadcast:
            for m in temp:
                if use_sparse_feedback:
                    W_above = self.W[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T

                if m == self.M - 2:
                    if noisy_symmetric_weights:
                        self.Y[m] = W_above + mat.random.normal(0, 0.05, size=W_above.shape)
                    else:
                        self.Y[m] = W_above
                else:
                    if noisy_symmetric_weights:
                        self.Y[m] = mat.dot(W_above + mat.random.normal(0, 0.05, size=W_above.shape), self.Y[m+1])
                    else:
                        self.Y[m] = mat.dot(W_above, self.Y[m+1])
        else:
            for m in xrange(self.M-2, -1, -1):
                if use_sparse_feedback:
                    W_above = self.W[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T

                if noisy_symmetric_weights:
                    self.Y[m] = W_above + mat.random.normal(0, 0.05)
                else:
                    self.Y[m] = W_above

    def init_layers(self):
        '''
        Create the layers of the network.
        '''
        self.l = list()
        if self.M <= 1:
            self.l.append(Guerguiev2017finalNode(net_model=self, m=-1, input_S=self.n_in))
        else:
            if use_broadcast:
                self.l.append(Guerguiev2017hiddenNode(model_net=self, m=0, input_S=self.n_in, b_input_S=self.n[-1]))
                for m_ in xrange(1, self.M-1):
                    self.l.append(Guerguiev2017hiddenNode(model_net=self, m=m_, input_S=self.n[m_-1], b_input_S=self.n[-1]))
            else:
                self.l.append(Guerguiev2017hiddenNode(model_net=self, m=0, input_S=self.n_in, b_input_S=self.n[1]))
                for m in xrange(1, self.M-1):
                    self.l.append(Guerguiev2017hiddenNode(model_net=self, m=m_, input_S=self.n[m_-1], b_input_S=self.n[m_+1]))
            self.l.append(Guerguiev2017finalNode(net_model=self, m=self.M-1, input_S=self.n[-2]))

    def record_zero_hists(self):
        if record_voltages:
            self.v_apical_hists = [ mat.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M-1) ]
            self.v_basal_hists = [ mat.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M) ]
            self.v_soma_hists = [ mat.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M) ]

    def record_csv(self):
        '''
        record data via training (hists for different position)
        '''
        if record_voltages:
            for layer_id in xrange(self.M):
                if layer_id != self.M-1:
                    with open(os.path.join(self.simulation_path, 'A_hist_{}.csv'.format(layer_id)), 'a') as A_hist_file:
                        mat.savetxt(A_hist_file, self.v_apical_hists[layer_id])
                with open(os.path.join(self.simulation_path, 'B_hist_{}.csv'.format(layer_id)), 'a') as B_hist_file:
                    mat.savetxt(B_hist_file, self.v_basal_hists[layer_id])
                with open(os.path.join(self.simulation_path, 'C_hist_{}.csv'.format(layer_id)), 'a') as C_hist_file:
                    mat.savetxt(C_hist_file, self.v_soma_hists[layer_id])
    
    def out_f(self, training=False):
        '''
        compute a forward phase pass through the network
        '''
        if self.M>1:
            rangeLayer=xrange(1, self.M-1)
        calculate_final_hist=self.l[-1].spike_hist
        calculate_final_rate=self.l[-1].spiking_rate

        if use_spiking_feedforward:
            x = self.x_hist
        else:
            x = self.x

        if self.M < 1:
            print("ERROR for out_f, layer less than lowest")
        elif self.M == 1:
            self.l[0].out_f(x, None)
        elif self.M > 1:
            if use_broadcast:
                if use_spiking_feedback:
                    self.l[0].out_f(x, calculate_final_hist)

                    for layer_id in rangeLayer:
                        if use_spiking_feedforward:
                            self.l[layer_id].out_f(self.l[layer_id-1].spike_hist, calculate_final_hist)
                        else:
                            self.l[layer_id].out_f(self.l[layer_id-1].spiking_rate, calculate_final_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].spike_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].spiking_rate, None)
                else:
                    self.l[0].out_f(x, calculate_final_rate)
                    for layer_id in rangeLayer:
                        if use_spiking_feedforward:
                            self.l[layer_id].out_f(self.l[layer_id-1].spike_hist, calculate_final_rate)
                        else:
                            self.l[layer_id].out_f(self.l[layer_id-1].spiking_rate, calculate_final_rate)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].spike_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].spiking_rate, None)
            else:
                if use_spiking_feedback:
                    self.l[0].out_f(x, self.l[1].spike_hist)

                    for layer_id in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[layer_id].out_f(self.l[layer_id-1].spike_hist, self.l[layer_id+1].spike_hist)
                        else:
                            self.l[layer_id].out_f(self.l[layer_id-1].spiking_rate, self.l[layer_id+1].spike_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].spike_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].spiking_rate, None)
                else:
                    self.l[0].out_f(x, self.l[1].spiking_rate)

                    for layer_id in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[layer_id].out_f(self.l[layer_id-1].spike_hist, self.l[layer_id+1].spiking_rate)
                        else:
                            self.l[layer_id].out_f(self.l[layer_id-1].spiking_rate, self.l[layer_id+1].spiking_rate)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].spike_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].spiking_rate, None)

    def out_t(self):
        '''
        compute a target phase pass for the network
        '''
        if self.M>1:
            rangeLayer=xrange(1, self.M-1)
        calculate_final_hist=self.l[-1].spike_hist
        calculate_final_rate=self.l[-1].spiking_rate
        if self.M>1:
            temp_range=xrange(1, self.M-1)

        if use_spiking_feedforward:
            x = self.x_hist
        else:
            x = self.x
        #print(type(x))
        #print(x.size())

        if self.M < 1:
            print("ERROR for out_t, layer less than lowest")
        elif self.M == 1:
            self.l[0].out_t(x, self.t)
        else:
            if use_broadcast:
                if use_spiking_feedback:
                    self.l[0].out_t(x, calculate_final_hist)

                    for layer_id in temp_range:
                        if use_spiking_feedforward:
                            self.l[layer_id].out_t(self.l[layer_id-1].spike_hist, calculate_final_hist)
                        else:
                            self.l[layer_id].out_t(self.l[layer_id-1].spiking_rate, calculate_final_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].spike_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].spiking_rate, self.t)
                else:
                    self.l[0].out_t(x, calculate_final_rate)

                    for layer_id in temp_range:
                        if use_spiking_feedforward:
                            self.l[layer_id].out_t(self.l[layer_id-1].spike_hist, calculate_final_rate)
                        else:
                            self.l[layer_id].out_t(self.l[layer_id-1].spiking_rate, calculate_final_rate)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].spike_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].spiking_rate, self.t)
            else:
                if use_spiking_feedback:
                    self.l[0].out_t(x, self.l[1].spike_hist)

                    for layer_id in temp_range:
                        if use_spiking_feedforward:
                            self.l[layer_id].out_t(self.l[layer_id-1].spike_hist, self.l[layer_id+1].spike_hist)
                        else:
                            self.l[layer_id].out_t(self.l[layer_id-1].spiking_rate, self.l[layer_id+1].spike_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].spike_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].spiking_rate, self.t)
                else:
                    self.l[0].out_t(x, self.l[1].spiking_rate)

                    for layer_id in temp_range:
                        if use_spiking_feedforward:
                            self.l[layer_id].out_t(self.l[layer_id-1].spike_hist, self.l[layer_id+1].spiking_rate)
                        else:
                            self.l[layer_id].out_t(self.l[layer_id-1].spiking_rate, self.l[layer_id+1].spiking_rate)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].spike_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].spiking_rate, self.t)

    def f_phase(self, x, t, training_num, training=False):
        '''
        compute a forward phase
        '''

        if training:
            self.record_zero_hists()

        for time in xrange(l_f_phase):

            pe = encoding.PoissonEncoder()
            temp_v = pe(torch.tensor(x))
            temp_v = mat.array(temp_v)
            self.x_hist = mat.concatenate([self.x_hist[:, 1:], temp_v], axis=-1)
            self.out_f(training=training)

            if training:
                if use_rand_plateau_times:
                    for layel_id in xrange(self.M-2, -1, -1):
                        plateau_indices = mat.nonzero(time == self.plateau_times_f[layel_id][training_num])
                        self.l[layel_id].plateau_f(plateau_indices=plateau_indices)

                if record_voltages:
                    for layel_id in xrange(self.M):
                        if layel_id != self.M-1:
                            self.v_apical_hists[m][time, :] = self.l[m].v_apical[:, 0]
                        self.v_basal_hists[m][time, :] = self.l[m].B[:, 0]
                        self.v_soma_hists[m][time, :] = self.l[m].C[:, 0]

        if not (use_rand_plateau_times and training):
            for layel_id in xrange(self.M-2, -1, -1):
                plateau_indices = mat.arange(self.n[layel_id])
                self.l[layel_id].plateau_f(plateau_indices=plateau_indices)
        for layel_id in xrange(self.M-1, -1, -1):
            self.l[layel_id].calc_averages(phase="forward")

        if record_eigvals:
            if len(self.J_betas) >= 100:
                self.J_betas = self.J_betas[1:]
                self.J_gammas = self.J_gammas[1:]

            self.J_betas.append(mat.multiply(lambda_max*deriv_sigma(self.l[-1].average_v_soma_f), k_D*self.W[-1]))
            self.J_gammas.append(mat.multiply(deriv_sigma(mat.dot(self.Y[-2], lambda_max*sigma(self.l[-1].average_v_soma_f))), self.Y[-2]))

        if training:
            self.record_csv()

    def clear_temp(self):
        for layer_id in xrange(self.M-1, -1, -1):
            self.l[layer_id].average_v_soma_f     *= 0
            self.l[layer_id].average_v_soma_t     *= 0
            self.l[layer_id].average_basal_input_f *= 0

            if layer_id == self.M-1:
                self.l[layer_id].average_spiking_rate_f *= 0
                self.l[layer_id].average_spiking_rate_t *= 0
            else:
                self.l[layer_id].average_v_apical_f     *= 0
                self.l[layer_id].average_v_apical_t     *= 0
                self.l[layer_id].average_spiking_rate_f *= 0
                if update_feedback_weights:
                    self.l[m].average_apical_input_f *= 0

    def t_phase(self, x, t, training_num):
        '''
        compute a target phase
        '''

        self.record_zero_hists()

        for time in xrange(l_t_phase):
            pe = encoding.PoissonEncoder()
            temp_v = pe(torch.tensor(x))
            temp_v = mat.array(temp_v)
            self.x_hist = mat.concatenate([self.x_hist[:, 1:], temp_v], axis=-1)

            calc_E_bp = record_backprop_angle and time == l_t_phase - 1
            self.out_t()

            #if taining:

            if use_rand_plateau_times:
                for layel_id in xrange(self.M-2, -1, -1):
                    plateau_indices = mat.nonzero(time == self.plateau_times_t[layel_id][training_num])
                    self.l[layel_id].plateau_t(plateau_indices=plateau_indices)

            if record_voltages:
                for layel_id in xrange(self.M):
                    if layel_id != self.M-1:
                        self.v_apical_hists[layel_id][time, :] = self.l[layel_id].v_apical[:, 0]
                    self.v_basal_hists[layel_id][time, :] = self.l[layel_id].v_basal[:, 0]
                    self.v_soma_hists[layel_id][time, :] = self.l[layel_id].v_soma[:, 0]

        if not use_rand_plateau_times:
            for layel_id in xrange(self.M-2, -1, -1):
                plateau_indices = mat.arange(self.n[layel_id])
                self.l[layel_id].plateau_t(plateau_indices=plateau_indices)

        for layel_id in xrange(self.M-1, -1, -1):
            self.l[layel_id].calc_averages(phase="target")
            if update_feedback_weights and layel_id < self.M-1:
                self.l[layel_id].update_Y()

            self.l[layel_id].update_W()

        #record loss in target processing
        if record_loss:
            pre=(self.l[-1].average_spiking_rate_t - lambda_max*sigma(self.l[-1].average_v_soma_f))
            self.loss = ( pre ** 2).mean()

        self.clear_temp()

        if use_symmetric_weights:
            self.make_weights_symmetric()

        if use_sparse_feedback and (use_symmetric_weights or update_feedback_weights):
            for m in xrange(self.M-1):
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0
                self.Y[m] *= 5

        self.record_csv()

    def train(self, f_lr, b_lr, n_epochs, n_training_examples, save_simulation, overwrite=False, simulation_notes=None, current_epoch=None, simulations_folder=default_simulations_folder, folder_name=""):
        '''
        Train the network. f_lr: feedforward learning rate; b_lr: feedback learning rate 
        '''

        print("Starting training.\n")

        if b_lr == None and update_feedback_weights:
            raise ValueError("No feedback learning rates.")

        if current_epoch != None:
            self.current_epoch == current_epoch
        elif self.current_epoch == None:
            self.current_epoch = 0
        continuing = (self.current_epoch != 0)

        if use_rand_phase_lengths:
            global l_f_phase, l_t_phase
            l_f_phases = min_l_f_phase + int(mat.random.wald(2, 1, n_training_examples * n_epochs)) # for all training examples
            l_t_phases = min_l_t_phase + int(random.wald(2, 1, n_training_examples * n_epochs))
        else:
            l_f_phases = mat.zeros(n_training_examples * n_epochs) + l_f_phase
            l_t_phases = mat.zeros(n_training_examples * n_epochs) + l_t_phase
        
        l_total = l_f_phases + l_t_phases

        self.f_lr = f_lr
        self.b_lr = b_lr

        if save_simulation:
            START_TIME = datetime.datetime.now()

            if folder_name == "":
                self.PATH = os.path.join(simulations_folder, "{}.{}.{}-{}.{}".format(START_TIME.year,START_TIME.month,START_TIME.day,START_TIME.hour,START_TIME.minute))
            else:
                self.PATH = os.path.join(simulations_folder, folder_name)

            if not os.path.exists(self.PATH):
                os.makedirs(self.PATH)
            elif not continuing:
                if not overwrite:
                    print("Error: Simulation directory \"{}\" already exists.".format(self.PATH))
                    return
                else:
                    shutil.rmtree(self.PATH, ignore_errors=True)
                    os.makedirs(self.PATH)

            filename = os.path.basename(__file__)
            if filename.endswith('pyc'):
                filename = filename[:-1]
            shutil.copyfile(filename, os.path.join(self.PATH, filename))

            params = {
                'nonspiking_mode'        : nonspiking_mode,
                'n_full_test'            : n_full_test,
                'n_quick_test'           : n_quick_test,
                'use_rand_phase_lengths' : use_rand_phase_lengths,
                'use_rand_plateau_times' : use_rand_plateau_times,
                'use_conductances'       : use_conductances,
                'use_broadcast'          : use_broadcast,
                'use_spiking_feedback'   : use_spiking_feedback,
                'use_spiking_feedforward': use_spiking_feedforward,
                'use_symmetric_weights'  : use_symmetric_weights,
                'noisy_symmetric_weights': noisy_symmetric_weights,
                'use_sparse_feedback'    : use_sparse_feedback,
                'update_feedback_weights': update_feedback_weights,
                'use_backprop'           : use_backprop,
                'use_apical_conductance' : use_apical_conductance,
                'use_weight_optimization': use_weight_optimization,
                'use_feedback_bias'      : use_feedback_bias,
                'initial_test'           : initial_test,
                'record_backprop_angle'  : record_backprop_angle,
                'record_loss'            : record_loss,
                'record_voltages'        : record_voltages,
                'record_training_error'  : record_training_error,
                'record_training_labels' : record_training_labels,
                'record_phase_times'     : record_phase_times,
                'record_plateau_times'   : record_plateau_times,
                'record_eigvals'         : record_eigvals,
                'record_matrices'        : record_matrices,
                'plot_eigvals'           : plot_eigvals,
                'dt'                     : dt,
                'mem'                    : mem,
                'integration_time'       : integration_time,
                'integration_time_test'  : integration_time_test,
                'l_f_phase'              : l_f_phase,
                'l_t_phase'              : l_t_phase,
                'l_f_phase_test'         : l_f_phase_test,
                'lambda_max'             : lambda_max,
                'tau_s'                  : tau_s,
                'tau_l'                  : tau_l,
                'g_b'                    : g_b,
                'g_a'                    : g_a,
                'g_l'                    : g_l,
                'g_d'                    : g_d,
                'k_B'                    : k_B,
                'k_D'                    : k_D,
                'k_I'                    : k_I,
                'P_hidden'               : P_hidden,
                'P_final'                : P_final,
                'n'                      : self.n,
                'f_lr'                   : f_lr,
                'b_lr'                   : b_lr,
                'n_training_examples'    : n_training_examples,
                'n_epochs'               : n_epochs
            }

            if continuing == False:
                with open(os.path.join(self.PATH, 'simulation.txt'), 'w') as sim_File:
                    print("Simulation done on {}.{}.{}-{}.{}.".format(START_TIME.year,START_TIME.month,START_TIME.day,START_TIME.hour,START_TIME.minute), file=sim_File)
                    if simulation_notes:
                        print(simulation_notes, file=sim_File)
                    print("Start time: {}".format(START_TIME), file=sim_File)
                    print("-----------------------------", file=sim_File)
                    for key, value in sorted(params.items()):
                        line = '{}: {}'.format(key, value)
                        print(line, file=sim_File)

                with open(os.path.join(self.PATH, 'simulation.json'), 'w') as sim_File:
                    sim_File.write(json.dumps(params))
            else:
                self.prev_full_test_errs = mat.load(os.path.join(self.PATH, "full_test_errors.npy"))
                self.prev_quick_test_errs = mat.load(os.path.join(self.PATH, "quick_test_errors.npy"))

                if record_backprop_angle and not use_backprop:
                    self.prev_bp_angles = mat.load(os.path.join(self.PATH, "bp_angles.npy"))

                if record_loss:
                    self.prev_losses = mat.load(os.path.join(self.PATH, "final_layer_loss.npy"))

                if record_training_error:
                    self.prev_training_errors = mat.load(os.path.join(self.PATH, "training_errors.npy"))

                if record_training_labels:
                    self.prev_training_labels = mat.load(os.path.join(self.PATH, "training_labels.npy"))

                if record_plateau_times:
                    self.prev_plateau_times_full = [ mat.load(os.path.join(self.PATH, "plateau_times_{}.npy".format(num))) for num in range(self.M)]

                if record_phase_times:
                    self.prev_phase_times = mat.load(os.path.join(self.PATH, "phase_times.npy"))

                if record_eigvals:
                    self.prev_max_jacobian_eigvals   = mat.load(os.path.join(self.PATH, "max_jacobian_eigvals.npy"))
                    self.prev_max_weight_eigvals     = mat.load(os.path.join(self.PATH, "max_weight_eigvals.npy"))
                    if record_matrices:
                        self.prev_jacobian_prod_matrices = mat.load(os.path.join(self.PATH, "jacobian_prod_matrices.npy"))
                        self.prev_weight_prod_matrices   = mat.load(os.path.join(self.PATH, "weight_prod_matrices.npy"))

        if save_simulation and not continuing:
            self.save_weights(self.PATH, prefix='initial_')

        if not continuing:
            self.full_test_errs  = mat.zeros(n_epochs + 1)
            self.quick_test_errs = mat.zeros(n_epochs*int(n_training_examples/1000.0) + 1)
        else:
            self.full_test_errs  = mat.zeros(n_epochs)
            self.quick_test_errs = mat.zeros(n_epochs*int(n_training_examples/1000.0))


        if record_loss:
            self.losses = mat.zeros(n_epochs*n_training_examples)

        if record_training_error:
            self.training_errors = mat.zeros(n_epochs)

        if record_plateau_times:
            self.plateau_times_full = [ mat.zeros((n_epochs*2*n_training_examples, self.n[m])) for m in range(self.M) ]

        if record_phase_times:
            self.phase_times = mat.zeros(n_epochs*n_training_examples*2)
            self.phase_times[0] = l_f_phases[0]
            for i in xrange(1, 2*n_epochs*n_training_examples):
                if i % 2 == 0:
                    self.phase_times[i] = self.phase_times[i-1] + l_f_phases[int(i/2)]
                else:
                    self.phase_times[i] = self.phase_times[i-1] + l_t_phases[int((i-1)/2)]

            if save_simulation:
                if not continuing:
                    phase_times = self.phase_times
                else:
                    phase_times = mat.concatenate(self.prev_phase_times, self.phase_times, axis=0)

                mat.save(os.path.join(self.PATH, "phase_times.npy"), phase_times)

        if record_training_labels:
            self.training_labels = mat.zeros(n_epochs*n_training_examples)

        if record_eigvals:
            self.max_jacobian_eigvals = mat.zeros(n_epochs*n_training_examples)
            if record_matrices:
                self.jacobian_prod_matrices = mat.zeros((n_epochs*n_training_examples, self.n[-1], self.n[-1]))

            if not continuing:
                self.max_weight_eigvals = mat.zeros(n_epochs*n_training_examples + 1)
                if record_matrices:
                    self.weight_prod_matrices = mat.zeros((n_epochs*n_training_examples + 1, self.n[-1], self.n[-1]))
            else:
                self.max_weight_eigvals = mat.zeros(n_epochs*n_training_examples)
                if record_matrices:
                    self.weight_prod_matrices = mat.zeros((n_epochs*n_training_examples, self.n[-1], self.n[-1]))

            I = mat.eye(self.n[-1])

            U = self.W[-1] @ self.Y[-2]
            p = (I - U).T @ (I - U)

            if not continuing:
                if record_matrices:
                    self.weight_prod_matrices[0] = U
                self.max_weight_eigvals[0] = mat.amax(mat.real(mat.linalg.eigvals(p)))

            self.J_betas = []
            self.J_gammas = []

        if record_backprop_angle and not use_backprop:
            if self.M > 1:
                self.bp_angles = mat.zeros(n_epochs*n_training_examples)

        if initial_test and not continuing:
            print("Start of epoch {}.".format(self.current_epoch + 1))

            start_time = time.time()

            test_err = self.test_weights(n_test=n_full_test)

            end_time = time.time()
            time_elapsed = end_time - start_time

            sys.stdout.write("\x1b[2K\rFE: {0:05.2f}%. T: {1:.3f}s.\n\n".format(test_err, time_elapsed))

            self.full_test_errs[0] = test_err

            if save_simulation:
                mat.save(os.path.join(self.PATH, "full_test_errors.npy"), self.full_test_errs)

                with open(os.path.join(self.PATH, "full_test_errors.txt"), 'a') as test_err_file:
                    line = "%.10f" % test_err
                    print(line, file=test_err_file)

            self.quick_test_errs[0] = test_err

            if save_simulation:
                # save quick test error
                mat.save(os.path.join(self.PATH, "quick_test_errors.npy"), self.quick_test_errs)

                with open(os.path.join(self.PATH, "quick_test_errors.txt"), 'a') as test_err_file:
                    line = "%.10f" % test_err
                    print(line, file=test_err_file)
        else:
            print("Start of epoch {}.\n".format(self.current_epoch + 1))

        # initialize input spike history
        self.x_hist = mat.zeros((self.n_in, mem))

        # start time used for timing how long each 1000 examples take
        start_time = None

        if record_training_error:
            num_correct = 0

        for k in xrange(n_epochs):
            self.x_train, self.t_train = shuffle_arrays(self.x_train, self.t_train)

            # generate arrays of forward & target phase plateau potential times
            if use_rand_plateau_times:
                self.plateau_times_f = [ mat.zeros((n_training_examples, self.n[m])) + l_f_phases[k*n_training_examples:(k+1)*n_training_examples, mat.newaxis] - 1 - mat.minimum(mat.abs(mat.random.normal(0, 3, size=(n_training_examples, self.n[m])).astype(int)), 5) for m in range(self.M) ]
                self.plateau_times_t = [ mat.zeros((n_training_examples, self.n[m])) + l_t_phases[k*n_training_examples:(k+1)*n_training_examples, mat.newaxis] - 1 - mat.minimum(mat.abs(mat.random.normal(0, 3, size=(n_training_examples, self.n[m])).astype(int)), 5) for m in range(self.M) ]
            else:
                self.plateau_times_f = [ mat.zeros((n_training_examples, self.n[m])) + l_f_phases[k*n_training_examples:(k+1)*n_training_examples, mat.newaxis] - 1 for m in range(self.M) ]
                self.plateau_times_t = [ mat.zeros((n_training_examples, self.n[m])) + l_t_phases[k*n_training_examples:(k+1)*n_training_examples, mat.newaxis] - 1 for m in range(self.M) ]

            for n in xrange(n_training_examples):
                if start_time == None:
                    start_time = time.time()

                if use_rand_phase_lengths:
                    l_f_phase = int(l_f_phases[k*n_training_examples + n])
                    l_t_phase = int(l_t_phases[k*n_training_examples + n])

                l_total = l_f_phase + l_t_phase

                if record_plateau_times:
                    total_time_to_forward_phase = mat.sum(l_f_phases[:k*n_training_examples + n]) + mat.sum(l_t_phases[:k*n_training_examples + n])
                    total_time_to_target_phase  = mat.sum(l_f_phases[:k*n_training_examples + n + 1]) + mat.sum(l_t_phases[:k*n_training_examples + n])
                    for m in range(self.M):
                        self.plateau_times_full[m][k*n_training_examples + 2*n]     = total_time_to_forward_phase + self.plateau_times_f[m][n]
                        self.plateau_times_full[m][k*n_training_examples + 2*n + 1] = total_time_to_target_phase + self.plateau_times_t[m][n]

                if (n+1) % 100 == 0:
                    sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}.".format(self.current_epoch + 1, n+1, n_training_examples))
                    sys.stdout.flush()

                self.x = lambda_max*self.x_train[:, n][:, mat.newaxis]
                self.t = self.t_train[:, n][:, mat.newaxis]

                if record_voltages:
                    self.v_apical_hists = [ mat.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M-1)]
                    self.v_basal_hists = [ mat.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M)]
                    self.v_soma_hists = [ mat.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M)]

                # forward phase
                self.f_phase(self.x, None, n, training=True)

                if record_training_error:
                    real = mat.argmax(mat.mean(self.l[-1].average_v_soma_f.reshape(-1, self.n_neurons_per_category), axis=-1))
                    target = mat.arange(10) @ self.t
                    num_correct += (real == target)
                
                # target phase
                self.t_phase(self.x, self.t.repeat(self.n_neurons_per_category, axis=0), n)

                if record_loss:
                    self.losses[k*n_training_examples + n] = self.loss

                if record_training_labels:
                    self.training_labels[k*n_training_examples + n] = mat.arange(10) @ self.t

                if record_eigvals:
                    U = mat.mean(mat.array([ self.J_betas[i] @ self.J_gammas[i] for i in range(len(self.J_betas)) ]), axis=0) # mean of product of last 100 Jacobians
                    
                    p = (I - U).T @ (I - U)
                    if record_matrices:
                        self.jacobian_prod_matrices[k*n_training_examples + n] = U
                    self.max_jacobian_eigvals[k*n_training_examples + n] = mat.amax(mat.linalg.eigvals(p))

                    U = k_D*self.W[-1] @ self.Y[-2]
                    p = (I - U).T @ (I - U)
                    if not continuing:
                        if record_matrices:
                            self.weight_prod_matrices[k*n_training_examples + n + 1] = U
                        self.max_weight_eigvals[k*n_training_examples + n + 1] = mat.amax(mat.linalg.eigvals(p))
                    else:
                        if record_matrices:
                            self.weight_prod_matrices[k*n_training_examples + n] = U
                        self.max_weight_eigvals[k*n_training_examples + n] = mat.amax(mat.linalg.eigvals(p))
                    
                    if plot_eigvals and k == 0 and n == 0:
                        # draw initial plots
                        if record_matrices:
                            A = self.jacobian_prod_matrices[0]
                            im_plot = ax1.imshow(A, interpolation='nearest', vmin=0, vmax=1)
                            fig.colorbar(im_plot, ax=ax1)
                        if record_loss:
                            loss_plot, = ax2.plot(mat.arange(1), self.losses[0])
                        max_jacobian_plot, = ax3.plot(mat.arange(1), self.max_jacobian_eigvals[0], '.')
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                if record_backprop_angle and not use_backprop:
                    # get backprop angle
                    if self.M > 1:
                        bp_angle = mat.arccos(mat.sum(self.l[0].delta_b_bp * self.l[0].delta_b_full) / (mat.linalg.norm(self.l[0].delta_b_bp)*mat.linalg.norm(self.l[0].delta_b_full.T)))*180.0/mat.pi
                        self.bp_angles[k*n_training_examples + n] = bp_angle

                if plot_eigvals and record_eigvals and (n+1) % 100 == 0:
                    max_inds = mat.argsort(self.max_jacobian_eigvals[k*n_training_examples + n -99:k*n_training_examples + n + 1])
                    max_ind = mat.argmax(self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1])
                    min_ind = mat.argmin(self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1])
                    n_small = mat.sum(self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1] < 1)
        
                    # update plots
                    if record_matrices:
                        A = mat.mean(mat.array([self.jacobian_prod_matrices[k*n_training_examples + n-99:k*n_training_examples + n + 1][i] for i in max_inds][:-10]), axis=0)
                        im_plot.set_data(A)

                    if record_loss:
                        loss_plot.set_xdata(mat.arange(k*n_training_examples + n))
                        loss_plot.set_ydata(self.losses[:k*n_training_examples + n])
                        ax2.set_xlim(0, k*n_training_examples + n)
                        ax2.set_ylim(mat.amin(self.losses[:k*n_training_examples + n]) - 1e-6, mat.amax(self.losses[:k*n_training_examples + n]) + 1e-6)

                    max_jacobian_plot.set_xdata(mat.arange(k*n_training_examples + n))
                    max_jacobian_plot.set_ydata(self.max_jacobian_eigvals[:k*n_training_examples + n])
                    ax3.set_xlim(0, k*n_training_examples + n)
                    ax3.set_ylim(mat.amin(self.max_jacobian_eigvals[:k*n_training_examples + n]) - 1e-6, mat.amax(self.max_jacobian_eigvals[:k*n_training_examples + n]) + 1e-6)

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if (n+1) % 1000 == 0:
                    if n != n_training_examples - 1:
                        # a quick weight test
                        test_err = self.test_weights(n_test=n_quick_test)

                        sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}. QE: {3:05.2f}%. ".format(self.current_epoch + 1, n+1, n_training_examples, test_err))

                        if not continuing:
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000)] = test_err
                        else:
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000) - 1] = test_err

                        if save_simulation:
                            with open(os.path.join(self.PATH, "quick_test_errors.txt"), 'a') as test_err_file:
                                line = "%.10f" % test_err
                                print(line, file=test_err_file)
                    else:
                        # a full weight test
                        test_err = self.test_weights(n_test=n_full_test)

                        sys.stdout.write("\x1b[2K\rFE: {0:05.2f}%. ".format(test_err))

                        if not continuing:
                            self.full_test_errs[k+1] = test_err
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000)] = test_err
                        else:
                            self.full_test_errs[k] = test_err
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000) - 1] = test_err

                        if save_simulation:
                            with open(os.path.join(self.PATH, "full_test_errors.txt"), 'a') as test_err_file:
                                line = "%.10f" % test_err
                                print(line, file=test_err_file)

                        if record_training_error:
                            err_rate = (1.0 - float(num_correct)/n_training_examples)*100.0
                            self.training_errors[k] = err_rate

                            print("Error rate: {0:05.2f}%. ".format(err_rate), end="")

                            num_correct = 0

                        # save recording arrays
                        if save_simulation:
                            print("Saving...", end="")
                            if not continuing:
                                # a new simulation
                                quick_test_errs = self.quick_test_errs[:(k+1)*int(n_training_examples/1000)+1]
                                full_test_errs  = self.full_test_errs[:k+2]

                                if record_backprop_angle and not use_backprop:
                                    bp_angles = self.bp_angles[:(k+1)*n_training_examples]

                                if record_loss:
                                    losses = self.losses[:(k+1)*n_training_examples]

                                if record_training_labels:
                                    training_labels = self.training_labels[:(k+1)*n_training_examples]

                                if record_plateau_times:
                                    plateau_times_full = [ self.plateau_times_full[m][:(k+1)*2*n_training_examples] for m in range(self.M) ]

                                if record_training_error:
                                    training_errors = self.training_errors[:k+1]

                                if record_eigvals:
                                    max_jacobian_eigvals   = self.max_jacobian_eigvals[:(k+1)*n_training_examples]
                                    max_weight_eigvals     = self.max_weight_eigvals[:(k+1)*n_training_examples+1]
                                    if record_matrices:
                                        jacobian_prod_matrices = self.jacobian_prod_matrices[:(k+1)*n_training_examples]
                                        weight_prod_matrices   = self.weight_prod_matrices[:(k+1)*n_training_examples+1]
                            else:
                                # a continuation of a previously-started simulation
                                quick_test_errs = mat.concatenate([self.prev_quick_test_errs, self.quick_test_errs[:(k+1)*int(n_training_examples/1000)]], axis=0)
                                if n == n_training_examples - 1:
                                    full_test_errs = mat.concatenate([self.prev_full_test_errs, self.full_test_errs[:k+1]], axis=0)

                                if record_backprop_angle and not use_backprop:
                                    bp_angles = mat.concatenate([self.prev_bp_angles, self.bp_angles[:(k+1)*n_training_examples]], axis=0)

                                if record_loss:
                                    losses = mat.concatenate([self.prev_losses, self.losses[:(k+1)*n_training_examples]], axis=0)

                                if record_training_labels:
                                    training_labels = mat.concatenate([self.prev_training_labels, self.training_labels[:(k+1)*n_training_examples]], axis=0)

                                if record_plateau_times:
                                    plateau_times_full = [ mat.concatenate([self.prev_plateau_times_full[m], self.plateau_times_full[m][:(k+1)*2*n_training_examples]]) for m in range(self.M) ]

                                if record_training_error:
                                    training_errors = mat.concatenate([self.prev_training_errors, self.training_errors[:k+1]], axis=0)

                                if record_eigvals:
                                    max_jacobian_eigvals   = mat.concatenate([self.prev_max_jacobian_eigvals, self.max_jacobian_eigvals[:(k+1)*n_training_examples]], axis=0)
                                    max_weight_eigvals     = mat.concatenate([self.prev_max_weight_eigvals, self.max_weight_eigvals[:(k+1)*n_training_examples]], axis=0)
                                    if record_matrices:
                                        jacobian_prod_matrices = mat.concatenate([self.prev_jacobian_prod_matrices, self.jacobian_prod_matrices[:(k+1)*n_training_examples]], axis=0)
                                        weight_prod_matrices   = mat.concatenate([self.prev_weight_prod_matrices, self.weight_prod_matrices[:(k+1)*n_training_examples]], axis=0)

                            mat.save(os.path.join(self.PATH, "quick_test_errors.npy"), quick_test_errs)

                            if n == n_training_examples - 1:
                                mat.save(os.path.join(self.PATH, "full_test_errors.npy"), full_test_errs)
                                self.save_weights(self.PATH, prefix="epoch_{}_".format(self.current_epoch + 1))

                            if record_backprop_angle and not use_backprop:
                                if self.M > 1:
                                    mat.save(os.path.join(self.PATH, "bp_angles.npy"), bp_angles)

                            if record_loss:
                                mat.save(os.path.join(self.PATH, "final_layer_loss.npy"), losses)

                            if record_training_labels:
                                mat.save(os.path.join(self.PATH, "training_labels.npy"), training_labels)

                            if record_plateau_times:
                                for m in range(self.M):
                                    mat.save(os.path.join(self.PATH, "plateau_times_{}.npy".format(m)), self.plateau_times_full[m])

                            if record_eigvals:
                                mat.save(os.path.join(self.PATH, "max_jacobian_eigvals.npy"), max_jacobian_eigvals)
                                mat.save(os.path.join(self.PATH, "max_weight_eigvals.npy"), max_weight_eigvals)
                                if record_matrices:
                                    mat.save(os.path.join(self.PATH, "jacobian_prod_matrices.npy"), jacobian_prod_matrices)
                                    mat.save(os.path.join(self.PATH, "weight_prod_matrices.npy"), weight_prod_matrices)

                            if record_training_error:
                                mat.save(os.path.join(self.PATH, "training_errors.npy"), training_errors)

                            print("done. ", end="")

                    if record_eigvals:
                        print("Min max Jacobian eigval: {:.4f}. ".format(mat.amin(self.max_jacobian_eigvals[max(0, k*n_training_examples + n - 999):k*n_training_examples + n + 1])), end="")
                        
                        print("# max eigvals < 1: {}. ".format(mat.sum(self.max_jacobian_eigvals[max(0, k*n_training_examples + n - 999):k*n_training_examples + n + 1] < 1)), end="")

                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    print("Time: {0:.3f}s.\n".format(time_elapsed))
                    start_time = None

            self.current_epoch += 1

        if save_simulation:
            with open(os.path.join(self.PATH, 'simulation.txt'), 'a') as sim_File:
                END_TIME = datetime.datetime.now()
                print("-----------------------------", file=sim_File)
                print("End time: {}".format(END_TIME), file=sim_File)

    def test_weights(self, n_test=n_quick_test):
        global l_f_phase, integration_time

        old_l_f_phase = l_f_phase
        l_f_phase = l_f_phase_test
        old_integration_time = integration_time
        integration_time = integration_time_test
        old_x_hist = self.x_hist
        num_correct = 0
        self.x_test, self.t_test = shuffle_arrays(self.x_test, self.t_test)

        digits = mat.arange(10)
        for m in xrange(self.M):
            self.l[m].create_integration_vars()

        for n in xrange(n_test):
            for m in xrange(self.M):
                self.l[m].origin()

            self.x_hist *= 0

            self.x = lambda_max*self.x_test[:, n][:, mat.newaxis]
            self.t = self.t_test[:, n][:, mat.newaxis]

            self.f_phase(self.x, self.t.repeat(self.n_neurons_per_category, axis=0), None, training=False)
            sel_num = mat.argmax(mat.mean(self.l[-1].average_v_soma_f.reshape(-1, self.n_neurons_per_category), axis=-1))

            target_num = mat.dot(digits, self.t)

            if sel_num == target_num:
                num_correct += 1

            if (n + 1) % 100  == 0:
                sys.stdout.write("\x1b[2K\rTesting example {0}/{1}. E: {2:05.2f}%.".format(n+1, n_test, (1.0 - float(num_correct)/(n+1))*100.0))
                sys.stdout.flush()

        err_rate = (1.0 - float(num_correct)/n_test)*100.0

        if old_x_hist is not None:
            self.x_hist = old_x_hist

        integration_time = old_integration_time

        l_f_phase = old_l_f_phase

        for m in xrange(self.M):
            self.l[m].create_integration_vars()

        for m in xrange(self.M):
            self.l[m].origin()

        if n_test > 100:
            sys.stdout.write("\x1b[2K\r")
            sys.stdout.flush()  

        return err_rate

    def save_weights(self, path, prefix=""):
        for m in xrange(self.M):
            mat.save(os.path.join(path, prefix + "W_{}.npy".format(m)), self.W[m])
            mat.save(os.path.join(path, prefix + "b_{}.npy".format(m)), self.b[m])
            if m != self.M - 1:
                mat.save(os.path.join(path, prefix + "Y_{}.npy".format(m)), self.Y[m])
                if use_feedback_bias:
                    mat.save(os.path.join(path, prefix + "c_{}.npy".format(m)), self.c[m])

    def load_weights(self, path, prefix=""):
        print("Loading weights from \"{}\" with prefix \"{}\".".format(path, prefix))

        for m in xrange(self.M):
            self.W[m] = mat.load(os.path.join(path, prefix + "W_{}.npy".format(m)))
            self.b[m] = mat.load(os.path.join(path, prefix + "b_{}.npy".format(m)))
            if m != self.M - 1:
                self.Y[m] = mat.load(os.path.join(path, prefix + "Y_{}.npy".format(m)))
                if use_feedback_bias:
                    self.c[m] = mat.load(os.path.join(path, prefix + "c_{}.npy".format(m)))

        self.print_weights()