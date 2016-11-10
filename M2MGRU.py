import theano
import theano.tensor as T
import lasagne
import numpy as np
from lasagne.regularization import l2

class M2MGRU(object):
    """
    A GRU neural network that has future contraint and can train/predict arbitrary length of sequence
    learning_rate
    drop_out: drop out rate
    N_hidden: number of nodes in each hidden dense layer
    D_input: dimension of input layer
    D_out: dimension of output layer
    Task_type: 'regression' or 'classification'
    L2_lambda: l2 regularization
    Layers: control the numbers of different layers [FF, GRU, FF], e.g. [2,1,1] means 2 feedforward layers, 1 GRU layer, 2 feedforward layers 

    """
    def __init__(self, learning_rate=1e-4, drop_out=0.3,Layers=[2,1,2], N_hidden=2048, N_cell=2048, D_input=39, D_out=120, Task_type='regression', L2_lambda=0.0, _EPSILON=1e-12):
        self.lr = learning_rate
        self.drop_out = drop_out
        self.N_hidden = N_hidden
        self.N_cell = N_cell
        self.D_input = D_input
        self.D_out = D_out
        self.Task_type = Task_type
        self.L2_lambda = L2_lambda

        #parameters of BGRU
        gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.001))
        cell = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            W_cell=None, b=lasagne.init.Constant(0.001),
            nonlinearity=lasagne.nonlinearities.tanh)

        #------varibles------
        #l2 regularization
        self.l2_penalty = 0
        #hard label
        self.hard_target = T.matrix('hard_target')
        #present sequence
        self.l_in_present = lasagne.layers.InputLayer(shape=(None, None, self.D_input))
        self.init_present = lasagne.layers.InputLayer(shape=(None, self.N_cell))
        n_batch, present_steps, n_features = self.l_in_present.input_var.shape
        #future sequence
        self.l_in_future = lasagne.layers.InputLayer(shape=(None, None, self.D_input)) 
        self.init_future = lasagne.layers.InputLayer(shape=(None, self.N_cell))
        n_batch, future_steps, n_features = self.l_in_future.input_var.shape

        #------networks------
        if Layers[0]!=0:
            #present sequence
            self.l_gru = lasagne.layers.ReshapeLayer(self.l_in_present, (-1, self.D_input))
            print('Reshape for Dense layers of present sequence',self.l_gru.output_shape)
            #future sequence
            self.l_gru_f = lasagne.layers.ReshapeLayer(self.l_in_future, (-1, self.D_input))
            print('Reshape for Dense layers of future sequence',self.l_gru_f.output_shape)
        else:
            self.l_gru = self.l_in_present
            self.l_gru_f = self.l_in_future

        #stack dense layers of present sequence
        for i in range(Layers[0]):
            self.l_gru= lasagne.layers.DenseLayer(
                self.l_gru,
                num_units=self.N_hidden, W=lasagne.init.HeUniform(gain='relu'),b=lasagne.init.Constant(0.001),
                nonlinearity=lasagne.nonlinearities.rectify)
            print('Present Dense %s' %i,self.l_gru.output_shape)
            self.l2_penalty += lasagne.regularization.regularize_layer_params(self.l_gru, l2) * L2_lambda
            self.l_gru=lasagne.layers.dropout(self.l_gru,self.drop_out)
            print('Dropout',self.drop_out)

        #stack dense layers of future sequence
        for i in range(Layers[0]):
            self.l_gru_f= lasagne.layers.DenseLayer(
                self.l_gru_f,
                num_units=self.N_hidden, W=lasagne.init.HeUniform(gain='relu'),b=lasagne.init.Constant(0.001),
                nonlinearity=lasagne.nonlinearities.rectify)
            print('Future Dense %s' %i,self.l_gru_f.output_shape)
            self.l2_penalty += lasagne.regularization.regularize_layer_params(self.l_gru_f, l2) * L2_lambda
            self.l_gru_f=lasagne.layers.dropout(self.l_gru_f,self.drop_out)
            print('Dropout',self.drop_out)

        if Layers[0]!=0:
            #reshape for present GRU
            self.l_gru = lasagne.layers.ReshapeLayer(self.l_gru, (-1, present_steps, self.N_hidden))
            print('Reshape for present GRU',self.l_gru.output_shape) 
            #reshape for future GRU
            self.l_gru_f = lasagne.layers.ReshapeLayer(self.l_gru_f, (-1, future_steps, self.N_hidden))
            print('Reshape for future GRU',self.l_gru_f.output_shape) 

        #M2MGRU
        for i in range(Layers[1]):
            if (i+1)!=Layers[1]:
                #present GRU 
                self.l_gru  =  lasagne.layers.recurrent.GRULayer(
                    self.l_gru, self.N_cell, grad_clipping=10.,
                    learn_init=0,resetgate=gate, updategate=gate, hidden_update=cell,backwards=False)
                #future GRU
                self.l_gru_f = lasagne.layers.recurrent.GRULayer(
                    self.l_gru_f, self.N_cell, grad_clipping=10.,
                    learn_init=0,resetgate=gate, updategate=gate, hidden_update=cell,backwards=True)
                self.l_hid = lasagne.layers.ElemwiseSumLayer([self.l_gru, self.l_gru_f])
                #merge
                print('Sum BGRU',self.l_hid.output_shape)
                self.l_hid=lasagne.layers.dropout(self.l_hid,self.drop_out)
                print('Dropout',self.drop_out)
                #for next GRU layers
                self.l_gru=self.l_hid
                self.l_gru_f=self.l_hid
                
            else:
                print('Last GRU')
                #present GRU 
                self.l_gru  =  lasagne.layers.recurrent.GRULayer(
                    self.l_gru, self.N_cell, grad_clipping=10.,
                    learn_init=0,resetgate=gate, updategate=gate, hidden_update=cell,
                    hid_init=self.init_present,backwards=False)
                #future GRU
                self.l_gru_f = lasagne.layers.recurrent.GRULayer(
                    self.l_gru_f, self.N_cell, grad_clipping=10.,
                    learn_init=0,resetgate=gate, updategate=gate, hidden_update=cell,
                    hid_init=self.init_future,backwards=True)
        #last hidden state of present GRU
        self.l_gru_l  =  lasagne.layers.SliceLayer(self.l_gru, indices=-1, axis=1)
        self.hid_present=lasagne.layers.get_output(self.l_gru_l, deterministic=True)  
        #last hidden state of present GRU
        self.l_gru_f_l  =  lasagne.layers.SliceLayer(self.l_gru_f, indices=-1, axis=1)
        self.hid_future = lasagne.layers.get_output(self.l_gru_f_l, deterministic=True)
        #merge present and future GRUs
        self.l_hid = lasagne.layers.ElemwiseSumLayer([self.l_gru, self.l_gru_f])
        print('merge present and future GRUs',self.l_hid.output_shape)
        self.l_hid=lasagne.layers.dropout(self.l_hid,self.drop_out)
        print('Dropout',self.drop_out)
        #reshape for Dense
        self.l_hid = lasagne.layers.ReshapeLayer(self.l_hid, (-1, self.N_cell))
        print('Reshape for Dense',self.l_hid.output_shape)
        #stack dense layers
        for i in range(Layers[2]):
            self.l_hid= lasagne.layers.DenseLayer(
                self.l_hid,
                num_units=self.N_hidden, W=lasagne.init.HeUniform(gain='relu'),b=lasagne.init.Constant(0.001),
                nonlinearity=lasagne.nonlinearities.rectify)
            print('Dense %s' %i,self.l_hid.output_shape)
            self.l2_penalty += lasagne.regularization.regularize_layer_params(self.l_hid, l2) * self.L2_lambda
            self.l_hid=lasagne.layers.dropout(self.l_hid,self.drop_out)
            print('Dropout',self.drop_out)
        #out_layer
        self.l_out=lasagne.layers.DenseLayer(self.l_hid, num_units=self.D_out, nonlinearity=lasagne.nonlinearities.linear)
        print('Output',self.l_out.output_shape)

        self.all_params = lasagne.layers.get_all_params(self.l_out)

        #------training function------
        #output of net for train / eval
        self.l_out_train = lasagne.layers.get_output(self.l_out, deterministic=False)
        self.l_out_eval = lasagne.layers.get_output(self.l_out, deterministic=True)
        if self.Task_type!='regression':
            self.l_out_train = T.exp(self.l_out_train)/T.sum(T.exp(self.l_out_train),axis=1, keepdims=True)
            self.l_out_eval = T.exp(self.l_out_eval)/T.sum(T.exp(self.l_out_eval),axis=1, keepdims=True)
            print('Softmax')
            self.l_out_train = T.clip(self.l_out_train, _EPSILON, 1.0 - _EPSILON)
            self.l_out_eval = T.clip(self.l_out_eval, _EPSILON, 1.0 - _EPSILON)

            self.loss_train = T.mean(lasagne.objectives.categorical_crossentropy(self.l_out_train, self.hard_target))
            self.loss_eval = T.mean(lasagne.objectives.categorical_crossentropy(self.l_out_eval, self.hard_target))
        else:
            print('Regression')
            self.loss_train = T.mean(lasagne.objectives.squared_error(self.l_out_train,self.hard_target))
            self.loss_eval = T.mean(lasagne.objectives.squared_error(self.l_out_eval,self.hard_target))
        #eval functions
        self.get_loss = theano.function([self.l_in_present.input_var,self.l_in_future.input_var, self.hard_target, self.init_present.input_var, self.init_future.input_var], self.loss_eval)
        self.updates = lasagne.updates.adam(self.loss_train + self.l2_penalty , self.all_params, learning_rate=self.lr)
        
        #train function
        self.train = theano.function([self.l_in_present.input_var,self.l_in_future.input_var, self.hard_target, self.init_present.input_var, self.init_future.input_var],[self.hid_present,self.hid_future], updates=self.updates)

        #output function
        self.get_out = theano.function([self.l_in_present.input_var,self.l_in_future.input_var, self.init_present.input_var, self.init_future.input_var],[self.l_out_eval, self.hid_present, self.hid_future])
       
    def saver(self,fpath):
        np.save(fpath,lasagne.layers.get_all_param_values(self.l_out))
    def loader(self,weights):
        lasagne.layers.set_all_param_values(self.l_out, weights)
