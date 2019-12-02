import tensorflow as tf

def SubsampleTDNN(_input, num_layers, layer_info, stddev=0.02):
    with tf.variable_scope('TDNN') as scope: 
        out_tdnn = _input 
        # calculate conv2d stride
        strides = []
        strides.append(layer_info[1].context[-1] - layer_info[1].context[0])
        for idx in range(1, num_layers):
            strides.append(2)

        # add conv2d ops for each TDNN layers
        for idx in range(num_layers):
            with tf.variable_scope('layer_' + str(idx)) as scope:
                # conv2 params
                out_channels  = layer_info[idx].num_filters
                in_channels   = tf.shape(out_tdnn)[3]
                filter_height = len(layer_info[idx].context)
                filter_width  = tf.shape(out_tdnn)[2]
                # define weight and bias tensor
                w = tf.get_variables(name = 'w', shape = [filter_height, filter_width, in_channels, out_channels], \
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                b = tf.get_variables(name = 'b', shape = [out_channels], \
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                # get output tensor of conv2d
                out_tdnn = tf.nn.conv2d(input = out_tdnn, filters = w, strides = [strides[idx], strides[idx]], \
                                  padding = 'VALID')
                # add bias
                out_tdnn = tf.nn.bias_add(out_tdnn, b)
                # add activation function
                out_tdnn = tf.nn.relu(out_tdnn)
    out_tdnn = tf.squeeze(out_tdnn)
    return out_tdnn
