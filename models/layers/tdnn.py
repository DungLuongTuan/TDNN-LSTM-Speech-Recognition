import tensorflow as tf


def SubsampleTDNN(_input, num_layers, layer_info, input_dim, max_left_frame, max_right_frame, stddev=0.02):
    with tf.variable_scope('SubsampleTDNN') as scope:
        # calculate list of first layer context index
        context_idxs = [max_left_frame]
        for idx in range(num_layers-1, -1, -1):
            new_context_idxs = []
            for upper_idx in context_idxs:
                for context in layer_info[idx].context:
                    new_context_idxs.append(upper_idx + context)
            context_idxs = new_context_idxs
        # get new input for hierachical tree
        out_tdnn = tf.gather(_input, context_idxs, axis = 1)
        # calculate in channels
        in_channels = []
        in_channels.append(1)
        for idx in range(num_layers - 1):
            in_channels.append(layer_info[idx].num_filters)
        # calculate filter widths
        filter_widths = []
        filter_widths.append(input_dim)
        for idx in range(1, num_layers):
            filter_widths.append(1)
        # add conv2d ops for each TDNN layer
        for idx in range(num_layers):
            with tf.variable_scope('layer_' + str(idx)) as scope:
                filter_height = len(layer_info[idx].context)
                filter_width  = filter_widths[idx]
                in_channel    = in_channels[idx]
                out_channel   = layer_info[idx].num_filters
                # define weight and bias tensor
                w = tf.get_variable(name = 'w', shape = [filter_height, filter_width, in_channel, out_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                b = tf.get_variable(name = 'b', shape = [out_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                # get output tensor of conv2d
                out_tdnn = tf.nn.conv2d(input = out_tdnn, filters = w, strides = [len(layer_info[idx].context), 1], padding = 'VALID')
                # add bias
                out_tdnn = tf.nn.bias_add(out_tdnn, b)
                # add activation function
                out_tdnn = tf.nn.relu(out_tdnn)
    out_tdnn = tf.squeeze(out_tdnn)
    return out_tdnn


def TDNN(_input, num_layers, layer_info, input_dim, subsample=False, stddev=0.02):
    with tf.variable_scope('TDNN') as scope:
        out_tdnn = _input
        # calculate in channels
        in_channels = []
        in_channels.append(1)
        for idx in range(num_layers - 1):
            in_channels.append(layer_info[idx].num_filters)
        # calculate filter widths
        filter_widths = []
        filter_widths.append(input_dim)
        for idx in range(1, num_layers):
            filter_widths.append(1)
        # add conv2d ops for each TDNN layer
        for idx in range(num_layers):
            with tf.variable_scope('layer_' + str(idx)) as scope:
                if  (idx == num_layers - 1) and (subsample == True):
                    l_f = tf.expand_dims(out_tdnn[:,0], 1)
                    r_f = tf.expand_dims(out_tdnn[:,-1], 1)
                    # batchsize x 2 x 1 x channels
                    out_tdnn = tf.concat([l_f, r_f], axis = 1)
                    filter_height = len(layer_info[idx].context)
                else:
                    filter_height = layer_info[idx].context[-1] - layer_info[idx].context[0] + 1
                # conv2 params
                out_channel  = layer_info[idx].num_filters
                in_channel   = in_channels[idx]
                filter_width  = filter_widths[idx]
                # define weight and bias tensor
                w = tf.get_variable(name = 'w', shape = [filter_height, filter_width, in_channel, out_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                b = tf.get_variable(name = 'b', shape = [out_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                # get output tensor of conv2d
                out_tdnn = tf.nn.conv2d(input = out_tdnn, filters = w, strides = [1, 1], padding = 'VALID')
                # add bias
                out_tdnn = tf.nn.bias_add(out_tdnn, b)
                # add activation function
                out_tdnn = tf.nn.relu(out_tdnn)
    out_tdnn = tf.squeeze(out_tdnn)
    return out_tdnn
