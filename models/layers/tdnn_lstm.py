import tensorflow as tf 
import numpy as np 
import pdb

def TDNN_LSTM(_input, num_layers, layer_info, input_dim, max_seqlen, max_left_frame, max_right_frame, stddev=0.02):
    with tf.variable_scope('TDNN_LSTM') as scope:
        # build hierachical context index of each layers
        context_idxs = list(np.arange(max_seqlen) + max_left_frame)
        for idx in range(num_layers-1, -1, -1):
            if layer_info[idx].layer_name == 'TDNN':
                new_context_idxs = []
                for upper_idx in context_idxs:
                    for context in layer_info[idx].context:
                        new_context_idxs.append(upper_idx + context)
                context_idxs = new_context_idxs

        # calculate filter widths
        filter_widths = []
        filter_widths.append(input_dim)
        for idx in range(num_layers - 1):
            if layer_info[idx].layer_name == 'TDNN':
                filter_widths.append(layer_info[idx].num_filters)
            elif layer_info[idx].layer_name == 'LSTM':
                filter_widths.append(layer_info[idx].num_units)

        # get new input follow context indexs
        output = tf.gather(_input, context_idxs, axis = 1)

        # add TDNN and LSTM layer
        for idx in range(num_layers):
            if layer_info[idx].layer_name == 'TDNN':
                with tf.variable_scope('layer_' + str(idx) + '-TDNN') as scope:
                    filter_height = len(layer_info[idx].context)
                    filter_width  = filter_widths[idx]
                    in_channel    = 1
                    out_channel   = layer_info[idx].num_filters
                    # define weight and bias tensor
                    w = tf.get_variable(name = 'w', shape = [filter_height, filter_width, in_channel, out_channel],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
                    b = tf.get_variable(name = 'b', shape = [out_channel],
                                        initializer=tf.truncated_normal_initializer(stddev=stddev))
                    # get output tensor of conv2d
                    output = tf.nn.conv2d(input = output, filters = w, strides = [len(layer_info[idx].context), 1], padding = 'VALID')
                    # add bias
                    output = tf.nn.bias_add(output, b)
                    # add activation function
                    output = tf.nn.relu(output)
                    output = tf.transpose(output, [0, 1, 3, 2])
            elif layer_info[idx].layer_name == 'LSTM':
                with tf.variable_scope('layer_' + str(idx) + '-LSTM') as scope:
                    output = tf.squeeze(output)
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units = layer_info[idx].num_units)
                    output, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = output)
                    output = tf.expand_dims(output, -1)
        # return output
        output = tf.squeeze(output)


