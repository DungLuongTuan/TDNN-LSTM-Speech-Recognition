import tensorflow as tf

def SubsampleTDNN(_input, num_layers, layer_info, input_dim, stddev=0.02):
    with tf.variable_scope('TDNN') as scope: 
        out_tdnn = _input 
        # calculate conv2d stride
        strides = []
        strides.append(layer_info[1].context[-1] - layer_info[1].context[0])
        for idx in range(1, num_layers):
            strides.append(2)
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
        # add conv2d ops for each TDNN layers
        for idx in range(num_layers):
            with tf.variable_scope('layer_' + str(idx)) as scope:
                # conv2 params
                out_channel  = layer_info[idx].num_filters
                in_channel   = in_channels[idx]
                filter_height = len(layer_info[idx].context)
                filter_width  = filter_widths[idx]
                # define weight and bias tensor
                w = tf.get_variable(name = 'w', shape = [filter_height, filter_width, in_channel, out_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                b = tf.get_variable(name = 'b', shape = [out_channel],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                # get output tensor of conv2d
                out_tdnn = tf.nn.conv2d(input = out_tdnn, filters = w, strides = [strides[idx], strides[idx]],
                                  padding = 'VALID')
                # add bias
                out_tdnn = tf.nn.bias_add(out_tdnn, b)
                # add activation function
                out_tdnn = tf.nn.relu(out_tdnn)
    out_tdnn = tf.squeeze(out_tdnn)
    return out_tdnn
