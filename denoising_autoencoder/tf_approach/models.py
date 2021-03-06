"""This file complements main.py, holding the definitions of the TF models that
   are passed to main's make_graph function.
"""

import tensorflow as tf



################################################################################
# DEFINE TENSORFLOW MODELS
################################################################################

# aliases
matmul = tf.matmul
relu = tf.nn.relu
dropout = tf.nn.dropout
l2loss = tf.nn.l2_loss
l2norm = tf.nn.l2_normalize
rfft = tf.spectral.rfft
conv1d = tf.nn.conv1d
conv2d = tf.nn.conv2d
max_pool = tf.nn.max_pool
batch_norm = tf.layers.batch_normalization

def weight_variable(shape, stddev=0.1, dtype=tf.float32):
    """Initializes a TF variable of 'shape', with truncated normal values of
       zero mean and given stddev.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=dtype))

def bias_variable(size, dtype=tf.float32):
    """Initializes a TF variable of rank 1 and length 'size' with values 0.1
    """
    return tf.Variable(tf.constant(0.1, shape=[size], dtype=dtype))

def simple_mlp(batch, num_classes, hidden_size=64):
    """A simple MLP. For every element of the input batch, performs:
       ========= LAYER 0 (input)==================================
                                         batch_size x chunk_size
       ========= LAYER 1 (hidden)=================================
       fully(chunk_sizexhidden_size)     batch_size x hidden_size
       relu
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


def deeper_mlp(batch, num_classes, hidden_size1=256, hidden_size2=128):
    """An MLP with 2 hidden layers:
       ========= LAYER 0 (input)==================================
                                         batch_size x chunk_size
       ========= LAYER 1 (hidden1)================================
       fully(chunk_sizexhidden_size1)    batch_size x hidden_size
       relu
       ========= LAYER 2 (hidden2)================================
       fully(chunk_sizexhidden_size2)    batch_size x hidden_size2
       relu
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




def concat_1d_with_fft(batch, norm_both=True):
    """If norm_both, performs concat(l2norm(batch),l2norm(abs(rfft(batch)))).
       For a batch of shape (a, b), the RFFT returns a shape of (a, b//2+1)
       of type complex64. The abs takes the magnitde of each as float32.
       The l2norm function forces each chunk of the batch to have a stddev=1.
       If norm_both=False, l2norm won't be called. Output shape=(a, b+(b//2+1)).
    """
    spectrum = tf.abs(rfft(batch))
    if norm_both:
        return tf.concat([l2norm(batch, dim=1), l2norm(spectrum, dim=1)], -1)
    else:
        return tf.concat([batch, spectrum], -1)

def fft_mlp(batch, num_classes, hidden_size=32):
    """This MLP takes into account the time-series wave together with
       the magnitudes of the RFFT of that wave.
       ========= LAYER 0 (input)==================================
       concat_1d_with_fft(batch, True) batch_size x chunk_size+(chunk_size//2+1)
       ========= LAYER 1 (hidden)=================================
       fully(chunk_size*1.5xhidden_size) batch_size x hidden_size
       relu
       ========= LAYER 2 (logits) ================================
       fully(hidden_size x num_classes ) batch_size x num_classes
    """
    batch_mix = concat_1d_with_fft(batch, norm_both=True)
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




def basic_convnet(batch, num_classes, patch_size=10, basenum_patches=20,
                  hidden_size=32):
    """A basic convnet with a minimal MLP on top:
       B:=batch_size, C:=chunk_size, P:=patch_size, K:=basenum_patches
       input:       B x C
       reshaped to  B x C x 1 x 1 (needed by conv2d)
       =========LAYER1========
       conv(P)      B x C-P+1     x K
       maxpool(2)   B x (C-P+1)/2 x K
       =========LAYER2========
       given L:=(C-P+1)/2
       conv(P)      B x L-P+1     x 2K
       maxpool(2)   B x (L-P+1)/2 x 2K
       reshaped to  B x (L-P+1)/2 *2*K
       =========FCN1========
       given M:=(L-P+1)/2 *2*K
       fully(M x hidden_size)           B x hidden_size
       =========TOP===========
       fully(hidden_size x num_classes) B x num_classes
    """
    batch_size, chunk_size = batch.get_shape().as_list()
    batch_expanded = tf.expand_dims(batch, -1)
    batch_expanded = tf.expand_dims(batch_expanded, -1)
    # conv layer 1
    W1 = weight_variable([patch_size, 1, 1, basenum_patches]) # h_w_cin_cout
    b1 = bias_variable(basenum_patches)
    conv1 = conv2d(batch_expanded, W1, [1, patch_size//2, 1, 1], "VALID")+b1
    pool1 = max_pool(conv1, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out1  = relu(pool1)
    # conv layer 2
    W2 = weight_variable([patch_size, 1, basenum_patches, basenum_patches*2])
    b2 = bias_variable(basenum_patches*2)
    conv2 = conv2d(out1, W2, [1,patch_size//2, 1, 1], "VALID")+b2
    pool2 = max_pool(conv2, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out2  = relu(pool2)
    conv_out = out2 # alias for the last conv layer
    # reshape last conv layer to (batch_size, h*w*c) to allow fully connected
    conv_shape = conv_out.get_shape().as_list() # [n, h, w, c]
    flatsize   = conv_shape[1]*conv_shape[2]*conv_shape[3] # h*w*c
    conv_flat  = tf.reshape(conv_out, [tf.shape(conv_out)[0], flatsize])
    # fully connected hidden layer:
    W_top1 = weight_variable([flatsize, hidden_size])
    b_top1 = bias_variable(hidden_size)
    top1   = relu(matmul(conv_flat, W_top1)+b_top1)
    # fully connected top layer
    W_top2 = weight_variable([hidden_size, num_classes])
    b_top2 = bias_variable(num_classes)
    logits = tf.matmul(top1, W_top2)+b_top2
    #
    l2reg = l2loss(W1)+l2loss(W1)+l2loss(W_top1)+l2loss(W_top2)
    return logits, l2reg



def deep_convnet(batch, num_classes, patch_size=3, basenum_patches=30,
                  hidden_size=64):
    """
    """
    batch_size, chunk_size = batch.get_shape().as_list()
    batch_expanded = tf.expand_dims(batch, -1)
    batch_expanded = tf.expand_dims(batch_expanded, -1)
    # conv layer 1
    W1 = weight_variable([patch_size, 1, 1, basenum_patches]) # h_w_cin_cout
    b1 = bias_variable(basenum_patches)
    conv1 = conv2d(batch_expanded, W1, [1, patch_size//2, 1, 1], "VALID")+b1
    pool1 = max_pool(conv1, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out1  = relu(pool1)
    # conv layer 2
    W2 = weight_variable([patch_size, 1, basenum_patches, basenum_patches*2])
    b2 = bias_variable(basenum_patches*2)
    conv2 = conv2d(out1, W2, [1, patch_size//2, 1, 1], "VALID")+b2
    pool2 = max_pool(conv2, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out2  = relu(pool2)
    # conv layer 3
    W3 = weight_variable([patch_size, 1, basenum_patches*2, basenum_patches*3])
    b3 = bias_variable(basenum_patches*3)
    conv3 = conv2d(out2, W3, [1, patch_size//2, 1, 1], "VALID")+b3
    pool3 = max_pool(conv3, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out3  = relu(pool3)
    # conv layer 4
    W4 = weight_variable([patch_size, 1, basenum_patches*3, basenum_patches*4])
    b4 = bias_variable(basenum_patches*4)
    conv4 = conv2d(out3, W4, [1, patch_size//2, 1, 1], "VALID")+b4
    pool4 = max_pool(conv4, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out4  = relu(pool4)
    # conv layer 5
    W5 = weight_variable([patch_size, 1, basenum_patches*4, basenum_patches*5])
    b5 = bias_variable(basenum_patches*5)
    conv5 = conv2d(out4, W5, [1, patch_size//2, 1, 1], "VALID")+b5
    pool5 = max_pool(conv5, [1,2,1,1], [1,2,1,1], "VALID") # ksize, strides, pad
    out5  = relu(pool5)
    conv_out = out5 # alias for the last conv layer
    # reshape last conv layer to (batch_size, h*w*c) to allow fully connected
    conv_shape = conv_out.get_shape().as_list() # [n, h, w, c]
    flatsize   = conv_shape[1]*conv_shape[2]*conv_shape[3] # h*w*c
    conv_flat  = tf.reshape(conv_out, [tf.shape(conv_out)[0], flatsize])
    # fully connected hidden layer:
    W_top1 = weight_variable([flatsize, hidden_size])
    b_top1 = bias_variable(hidden_size)
    top1   = relu(matmul(conv_flat, W_top1)+b_top1)
    # fully connected top layer
    W_top2 = weight_variable([hidden_size, num_classes])
    b_top2 = bias_variable(num_classes)
    logits = tf.matmul(top1, W_top2)+b_top2
    #
    l2reg = l2loss(W1)+l2loss(W1)+l2loss(W_top1)+l2loss(W_top2)
    return logits, l2reg
