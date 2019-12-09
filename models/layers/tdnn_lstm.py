import tensorflow as tf 
import numpy as np 
import pdb

# def TDNN_LSTM(_input, num_layers, layer_info, input_dim, max_seqlen, max_left_frame, max_right_frame, stddev=0.02):
#     with tf.variable_scope('TDNN_LSTM') as scope:
#         # build hierachical context index of each layers
#         context_idxs = list(np.arange(max_seqlen) + max_left_frame)
#         for idx in range(num_layers-1, -1, -1):
#             if layer_info[idx].layer_name == 'TDNN':
#                 new_context_idxs = []
#                 for upper_idx in context_idxs:
#                     for context in layer_info[idx].context:
#                         new_context_idxs.append(upper_idx + context)
#                 context_idxs = new_context_idxs

#         # calculate filter widths
#         filter_widths = []
#         filter_widths.append(input_dim)
#         for idx in range(num_layers - 1):
#             if layer_info[idx].layer_name == 'TDNN':
#                 filter_widths.append(layer_info[idx].num_filters)
#             elif layer_info[idx].layer_name == 'LSTM':
#                 filter_widths.append(layer_info[idx].num_units)

#         # get new input follow context indexs
#         output = tf.gather(_input, context_idxs, axis = 1)

#         # add TDNN and LSTM layer
#         for idx in range(num_layers):
#             if layer_info[idx].layer_name == 'TDNN':
#                 with tf.variable_scope('layer_' + str(idx) + '-TDNN') as scope:
#                     filter_height = len(layer_info[idx].context)
#                     filter_width  = filter_widths[idx]
#                     in_channel    = 1
#                     out_channel   = layer_info[idx].num_filters
#                     # define weight and bias tensor
#                     w = tf.get_variable(name = 'w', shape = [filter_height, filter_width, in_channel, out_channel],
#                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
#                     b = tf.get_variable(name = 'b', shape = [out_channel],
#                                         initializer=tf.truncated_normal_initializer(stddev=stddev))
#                     # get output tensor of conv2d
#                     output = tf.nn.conv2d(input = output, filters = w, strides = [len(layer_info[idx].context), 1], padding = 'VALID')
#                     # add bias
#                     output = tf.nn.bias_add(output, b)
#                     # add activation function
#                     output = tf.nn.relu(output)
#                     output = tf.transpose(output, [0, 1, 3, 2])
#             elif layer_info[idx].layer_name == 'LSTM':
#                 with tf.variable_scope('layer_' + str(idx) + '-LSTM') as scope:
#                     # tf.squeeze return unknown shape and cannot feed to dynamic_rnn, use other ops instead          
#                     # output = tf.squeeze(output)
#                     output = tf.transpose(output, [3, 0, 1, 2])[0]
#                     lstm_cell = tf.contrib.rnn.LSTMCell(num_units = layer_info[idx].num_units)
#                     output, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = output, dtype=tf.float32)
#                     output = tf.expand_dims(output, -1)
#         # return output
#         output = tf.squeeze(output)
#         return output


def TDNN_LSTM(_input, sequence_length, num_layers, layer_info, input_dim, max_seqlen, max_left_frame, max_right_frame, stddev=0.02):
    with tf.variable_scope('TDNN_LSTM') as scope:
        # build hierachical context index of each layers
        upper_layer_idxs = list(np.arange(max_seqlen) + max_left_frame)
        layers_context = []
        for lstm_idx in range(num_layers-1, -1, -1):
            context_idxs = upper_layer_idxs
            for tdnn_idx in range(layer_info[lstm_idx].tdnn_num_layers-1, -1, -1):
                new_context_idxs = []
                for upper_idx in context_idxs:
                    for context in layer_info[lstm_idx].tdnn_layer_info[tdnn_idx].context:
                        new_context_idxs.append(upper_idx + context)
                context_idxs = new_context_idxs
            layers_context.append(context_idxs)
            upper_layer_idxs = list(np.arange(max(context_idxs) - min(context_idxs) + 1) + min(context_idxs))
        layers_context.reverse()

        # calcualte input_dim of each lstm layer
        input_dims = []
        input_dims.append(input_dim)
        for lstm_idx in range(num_layers - 1):
            input_dims.append(layer_info[lstm_idx].lstm_num_units)

        # add TDNN LSTM layers
        output = _input
        for lstm_idx in range(num_layers):
            with tf.variable_scope('TDNN_LSTM_layer_' + str(lstm_idx)) as scope:
                # calculate filter widths
                filter_widths = []
                filter_widths.append(input_dims[lstm_idx])
                for tdnn_idx in range(layer_info[lstm_idx].tdnn_num_layers - 1):
                    filter_widths.append(layer_info[lstm_idx].tdnn_layer_info[tdnn_idx].num_filters)

                # get new input of current lstm layer follow context indexs of upper lstm layer
                output = tf.gather(output, layers_context[lstm_idx], axis = 1)

                # add TDNN layers
                for tdnn_idx in range(layer_info[lstm_idx].tdnn_num_layers):
                    with tf.variable_scope('TDNN_layer_' + str(tdnn_idx)) as scope:
                        filter_height = len(layer_info[lstm_idx].tdnn_layer_info[tdnn_idx].context)
                        filter_width  = filter_widths[tdnn_idx]
                        in_channel    = 1
                        out_channel   = layer_info[lstm_idx].tdnn_layer_info[tdnn_idx].num_filters
                        # define weight and bias tensor
                        w = tf.get_variable(name = 'w', shape = [filter_height, filter_width, in_channel, out_channel],
                                            initializer=tf.truncated_normal_initializer(stddev=stddev))
                        b = tf.get_variable(name = 'b', shape = [out_channel],
                                            initializer=tf.truncated_normal_initializer(stddev=stddev))
                        # get output tensor of conv2d
                        output = tf.nn.conv2d(input = output, filters = w, strides = [len(layer_info[lstm_idx].tdnn_layer_info[tdnn_idx].context), 1], padding = 'VALID')
                        # add bias
                        output = tf.nn.bias_add(output, b)
                        # add activation function
                        output = tf.nn.relu(output)
                        output = tf.transpose(output, [0, 1, 3, 2])
                # add LSTM layer
                with tf.variable_scope('LSTM_layer') as scope:
                    output = tf.transpose(output, [3, 0, 1, 2])[0]
                    lstm_cell = tf.contrib.rnn.LSTMCell(num_units = layer_info[lstm_idx].lstm_num_units)
                    if lstm_idx == num_layers - 1:
                        output, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = output, dtype=tf.float32, sequence_length = sequence_length)
                    else:
                        output, state = tf.nn.dynamic_rnn(cell = lstm_cell, inputs = output, dtype=tf.float32, sequence_length = None)
                    output = tf.expand_dims(output, -1)
    # return output
    output = tf.squeeze(output)
    return output
