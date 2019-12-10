import tensorflow as tf 

def Softmax(_input, input_dim, output_dim, stddev = 0.02):
	# define initializer
    # sqrt3 = math.sqrt(3)
    # initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype = tf.float64)
    initializer = tf.contrib.layers.xavier_initializer()
	
    with tf.name_scope("Softmax"):
        w = tf.get_variable(name = 'w', shape = [input_dim, output_dim], \
                            initializer=initializer)
  
        b = tf.get_variable(shape = [1, output_dim], name = 'b', \
                            initializer=initializer)
        output = tf.nn.softmax(tf.matmul(_input, w) + b)
        return output