
def Softmax(_input, output_dim, stddev = 0.02):
    with tf.name_scope("Softmax"):
        w = tf.get_variable(name = 'w', shape = [tf.shape(_input)[-1], output_dim], \
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
  
        b = tf.get_variable(shape = [1, output_dim], name = 'b', \
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        output = tf.nn.softmax(tf.matmul(output_slice, w) + b)
        return output