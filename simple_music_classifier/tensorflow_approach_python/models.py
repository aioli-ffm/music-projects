import tensorflow as tf




################################################################################
# DEFINE TENSORFLOW MODELS
################################################################################

# aliases
matmul = tf.matmul
relu = tf.nn.relu
dropout = tf.nn.dropout
l2loss = tf.nn.l2_loss

# conv1d = tf.nn.conv1d
# conv2d = tf.nn.conv2d
# max_pool = tf.nn.max_pool
# batch_norm = tf.layers.batch_normalization

def weight_variable(shape, stddev=0.001, dtype=tf.float32):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=dtype))

def bias_variable(shape, dtype=tf.float32):
    return tf.Variable(tf.constant(0.1, shape=[shape], dtype=dtype))

def simple_mlp(batch, num_classes, hidden_size=64):
    """A simple MLP. For every element of the input batch, performs:
       ========= LAYER 0 (input)==================================
                                         batch_size x chunk_size
       ========= LAYER 1 (hidden)=================================
       fully(chunk_sizexhidden_size)     batch_size x hidden_size
       ========= LAYER 2 (logits) ================================
       fully(hidden_size x num_classes ) batch_size x num_classes
    """
    batch_size, chunk_size = batch.get_shape().as_list()
    #
    W1 = weight_variable([chunk_size, hidden_size], dtype=tf.float32)
    b1 = bias_variable(hidden_size, dtype=tf.float32)
    out1 = relu(matmul(batch, W1)+b1)
    #
    W2 = weight_variable([hidden_size, num_classes], dtype=tf.float32)
    b2 = bias_variable(num_classes, dtype=tf.float32)
    logits = matmul(out1, W2) + b2
    return logits, l2loss(W1)+l2loss(W2)


def deep_mlp(batch, num_classes, hidden_size1=256, hidden_size2=128):
    """An MLP with 2 hidden layers:
       ========= LAYER 0 (input)==================================
                                         batch_size x chunk_size
       ========= LAYER 1 (hidden1)================================
       fully(chunk_sizexhidden_size1)    batch_size x hidden_size
       ========= LAYER 2 (hidden2)================================
       fully(chunk_sizexhidden_size2)    batch_size x hidden_size2
       ========= LAYER 3 (logits) ================================
       fully(hidden_size2 x num_classes ) batch_size x num_classes
    """
    batch_size, chunk_size = batch.get_shape().as_list()
    #
    W1 = weight_variable([chunk_size, hidden_size1], dtype=tf.float32)
    b1 = bias_variable(hidden_size1, dtype=tf.float32)
    out1 = relu(matmul(batch, W1)+b1)
    #
    W2 = weight_variable([hidden_size1, hidden_size2], dtype=tf.float32)
    b2 = bias_variable(hidden_size2, dtype=tf.float32)
    out2 = relu(matmul(out1, W2)+b2)
    #
    W3 = weight_variable([hidden_size2, num_classes], dtype=tf.float32)
    b3 = bias_variable(num_classes, dtype=tf.float32)
    logits = matmul(out2, W3) + b3
    return logits, l2loss(W1)+l2loss(W2)+l2loss(W3)




def fft_mlp(batch, num_classes, hidden_size=256):
    """This MLP takes into account the time-series wave together with
       the magnitudes of the RFFT of that wave.
    """
    batch_fft_magnitudes = tf.abs(tf.spectral.rfft(batch, name="fft"))
    batch_mix = tf.concat([tf.nn.l2_normalize(batch,dim=-1),
                           tf.nn.l2_normalize(batch_fft_magnitudes,dim=-1)], -1)
    batch_size, chunk_size = batch_mix.get_shape().as_list()
    #
    W1 = weight_variable([chunk_size, hidden_size], dtype=tf.float32)
    b1 = bias_variable(hidden_size, dtype=tf.float32)
    out1 = relu(matmul(batch_mix, W1)+b1)
    #
    W2 = weight_variable([hidden_size, num_classes], dtype=tf.float32)
    b2 = bias_variable(num_classes, dtype=tf.float32)
    logits = matmul(out1, W2) + b2
    return logits, l2loss(W1)+l2loss(W2)
