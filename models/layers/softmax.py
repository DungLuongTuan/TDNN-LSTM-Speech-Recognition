import tensorflow as tf 

def Softmax(_input, input_dim, output_dim, stddev = 0.02):
    with tf.name_scope("Softmax"):
        w = tf.get_variable(name = 'w', shape = [input_dim, output_dim], \
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
  
        b = tf.get_variable(shape = [1, output_dim], name = 'b', \
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        output = tf.nn.softmax(tf.matmul(_input, w) + b)
        return output