# Rout, Biraaj
# 1002_071_886
# 2023_03_19
# Assignment_02_01

import numpy as np
import tensorflow as tf

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                              validation_split=[0.8,1.0],weights=None,seed=2):
    # This function creates and trains a multi-layer neural Network
    # X_train: numpy array of input for training [nof_train_samples,input_dimensions]
    # Y_train: numpy array of desired outputs for training samples [nof_train_samples,output_dimensions]
    # layers: list of integers representing number of nodes in each layer
    # activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
    # are, "linear", "sigmoid", "relu".
    # alpha: learning rate
    # epochs: number of epochs for training.
    # loss: is a case-insensitive string determining the loss function. The possible inputs are: "svm" , "mse",
    # "cross_entropy". for cross entropy use the tf.nn.softmax_cross_entropy_with_logits().
    # validation_split: a two-element list specifying the normalized start and end point to
    # extract validation set. Use floor in case of non integers.
    # weights: list of numpy weight matrices. If weights is equal to None then it should be ignored, otherwise,
    # the weight matrices should be initialized by the values given in the weight list (no random
    # initialization when weights is not equal to None).
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list should be a 2-d numpy array which corresponds to the weight matrix of the
        # corresponding layer.

        # The second element should be a one dimensional numpy array of numbers
        # representing the error after each epoch. Each error should
        # be calculated by using the validation set while the network is frozen.
        # Frozen means that the weights should not be adjusted while calculating the error.

        # The third element should be a two-dimensional numpy array [nof_validation_samples,output_dimensions]
        # representing the actual output of the network when validation set is used as input.

    # Notes:
    # The data set in this assignment is the transpose of the data set in assignment_01. i.e., each row represents
    # one data sample.
    # The weights in this assignment are the transpose of the weights in assignment_01.
    # Each output weights in this assignment is the transpose of the output weights in assignment_01
    # DO NOT use any other package other than tensorflow and numpy
    # Bias should be included in the weight matrix in the first row.
    # Use steepest descent for adjusting the weights
    # Use minibatch to calculate error and adjusting the weights
    # Reseed the random number generator when initializing weights for each layer.
    # Use numpy for weight to initialize weights. Do not use tensorflow weight initialization.
    # Do not use any random method from tensorflow
    # Do not shuffle data
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    
    #weight intialization
    nn_weights = []
    if weights == None:
        nn_weights = network_init(X_train.shape[1],layers,seed)
    else:
        nn_weights = weights
    
    tensor_weights = [tf.convert_to_tensor(_weight,dtype=tf.float32) for _weight in nn_weights]
    
    #Split data before running minibatch
    split_x_train,split_y_train,split_x_test,split_y_test = split_data(X_train,Y_train,validation_split)
    _column_addition = [[0,0],[1,0]]
    split_x_test = tf.convert_to_tensor(split_x_test,dtype=tf.float32)
    split_y_test = tf.convert_to_tensor(split_y_test,dtype=tf.float32)
    
    #Error List
    error_list = []
    final_output_matrix = []
    
    #Run epochs
    if epochs != 0:
        for epoch in range(epochs):
            for batch_x_train,batch_y_train in generate_batches(split_x_train,split_y_train,batch_size):
                with tf.GradientTape() as tape:
                    tape.watch(tensor_weights)
                    batch_x_train = tf.convert_to_tensor(batch_x_train,dtype=tf.float32)
                    batch_y_train = tf.convert_to_tensor(batch_y_train,dtype=tf.float32)
                    predictions = None
                    activated_output = None
                    error = None
                    prev_forward_propagate_output = batch_x_train
                    for _index,_weight in enumerate(tensor_weights):
                        weight_tensor = tensor_weights[_index]
                        #Adding bias after each layer output
                        forward_propagate_output = forward_propagation(tf.pad(prev_forward_propagate_output,_column_addition,constant_values=1),weight_tensor)
                        activated_output = activation(activations[_index],forward_propagate_output)
                        prev_forward_propagate_output = activated_output
                    
                    predictions = activated_output
                    error = _loss(loss,batch_y_train,predictions)
                    
                #Backward Propagation
                _gradient = tape.gradient(error,tensor_weights)
                
                for _index,_weight in enumerate(tensor_weights):
                    tensor_weights[_index] = tensor_weights[_index] - alpha*_gradient[_index]
                    
            
            #Compute validation loss for each epoch
            activated_test_output = None
            test_error = None
            prev_forward_propagate_test_output = split_x_test
            for _index,_weight in enumerate(tensor_weights):
                #Adding bias after each layer output
                forward_propagate_test_output = forward_propagation(tf.pad(prev_forward_propagate_test_output,_column_addition,constant_values=1),tensor_weights[_index])
                activated_test_output = activation(activations[_index],forward_propagate_test_output)
                prev_forward_propagate_test_output = activated_test_output
            test_error = _loss(loss,split_y_test,activated_test_output)
            error_list.append(float(test_error))
        
            
    nn_weights = [_tensor.numpy() for _tensor in tensor_weights]
    
    # Final Output after training or when epochs=0
    prev_forward_propagate_final_output = split_x_test
    for _index,_weight in enumerate(tensor_weights):
        #Adding bias after each layer output
        forward_propagate_test_output = forward_propagation(tf.pad(prev_forward_propagate_final_output,_column_addition,constant_values=1),tensor_weights[_index])
        final_output_matrix = activation(activations[_index],forward_propagate_test_output)
        prev_forward_propagate_final_output = final_output_matrix
    

    return [nn_weights,error_list,final_output_matrix.numpy()]

def network_init(input_dimension,layers,seed):
    #This function intializes weight
    #This function is taken from assignment 1 with change in intializations dimensions.
    nn_weights = []
    np.random.seed(seed)
    first_weight_layer = np.random.randn(input_dimension+1,layers[0])
    nn_weights.append(first_weight_layer)
    prev_dimension = layers[0]
    if len(layers) > 1:
        for nodes in layers[1:]:
            np.random.seed(seed)
            nn_weights.append(np.random.randn(prev_dimension+1,nodes))
            prev_dimension = nodes
    return nn_weights

def forward_propagation(_features,_weight):
    return tf.matmul(_features, _weight)

#This function calculates the activation of nn after forward propagation.
def activation(_function,value):
    #activation  are, "linear", "sigmoid", "relu".
    if _function == "linear":
        return value
    if _function == "sigmoid":
        return 1 / (1 + tf.exp(-value))
    if _function == "relu":
        return tf.maximum(0.0, value)

#This function calculates loss of nn after activation
def _loss(_function,_original,_predicted):
    #losses are: "svm" , "mse", "cross_entropy"
    if _function == "svm":
        return tf.reduce_sum(tf.maximum(0.0,(1.0-tf.multiply(_original,_predicted))))
    if _function == "mse":
        return tf.reduce_mean(tf.square(_original-_predicted))
    if _function == "cross_entropy":
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_original,_predicted))
        
#The below functions were used from helper.py
def split_data(X_train, Y_train, split_range=[0.2, 0.7]):
    start = int(split_range[0] * X_train.shape[0])
    end = int(split_range[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]

def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    # if there's any data left, yield it
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]