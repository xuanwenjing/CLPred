import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
bool_list_correct_prediction = tf.placeholder(tf.float32)
# %%
def conv(layer_name, x, out_channels, kernel_size=[3, 3], padding = 'SAME',stride=[1, 1, 1, 1], trainable=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding=padding, name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x
def pool(layer_name, x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1],padding='SAME', is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding=padding, name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding=padding, name=layer_name)
    return x
def conv2D(layer_name, x, out_channels, trainable,kernel_size=[3, 3], stride=[1, 1, 1, 1],padding='SAME',activation_fn=True,l2_value=0.00004):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable= (trainable is not None),
                            regularizer=tf.contrib.layers.l2_regularizer(l2_value),
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=(trainable is not None),
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding=padding, name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        if activation_fn:
            x = tf.nn.relu(x, name='relu')
        return x
def conv1D(layer_name, x, out_channels, kernel_size=3, stride=1,padding='SAME',trainable=True,activation_fn=True,l2_value=0.00004):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers.
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            regularizer=tf.contrib.layers.l2_regularizer(l2_value),
                            shape=[kernel_size, in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv1d(x, w, stride, padding=padding, name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        if activation_fn:
            x = tf.nn.relu(x, name='relu')
        return x
def conv2D_norm(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1],padding='SAME',trainable=True,is_training=True,activation_fn=True,l2_value=0.00004):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        trainable: if load pretrained parameters, freeze all conv layers.
        is_training: if is training,if so ,set True
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            regularizer=tf.contrib.layers.l2_regularizer(l2_value),
                            trainable=trainable,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        x = tf.nn.conv2d(x, w, stride, padding=padding, name='conv')
        with tf.variable_scope("BatchNorm"):
            x = tf.layers.batch_normalization(x,training=is_training,momentum=0.9)
        if activation_fn:
            x = tf.nn.relu(x, name='relu')
        return x
def conv1D_norm(layer_name, x, out_channels, is_training,kernel_size=3, stride=1,padding='SAME',trainable=True,activation_fn=True,l2_value=0.00004):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        trainable: if load pretrained parameters, freeze all conv layers.
        is_training: if is training,if so ,set True
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=trainable,
                            regularizer=tf.contrib.layers.l2_regularizer(l2_value),
                            shape=[kernel_size, in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=trainable,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv1d(x, w, stride, padding=padding, name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        #x = batch_norm_wrapper(x, is_training)
        with tf.variable_scope("BatchNorm"):
            x = tf.layers.batch_normalization(x,training=is_training,momentum=0.9)
        if activation_fn:
            x = tf.nn.relu(x, name='relu')
        return x
def conv_norm(layer_name, x, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1],padding='SAME',trainable=True,is_training=True,activation_fn=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        trainable: if load pretrained parameters, freeze all conv layers.
        is_training: if is training,if so ,set True
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            regularizer=tf.contrib.layers.l2_regularizer(0.00004),
                            trainable=(trainable is not None),
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())  # default is uniform distribution initialization
        x = tf.nn.conv2d(x, w, stride, padding=padding, name='conv')
        with tf.variable_scope("BatchNorm"):
            x = batch_norm_wrapper(x,is_training)
        if activation_fn:
            x = tf.nn.relu(x, name='relu')
        return x

def batch_norm_wrapper(inputs, is_training, decay = 0.9997):
    epsilon = 0.001
    scale = tf.get_variable(name='scale',shape=[inputs.get_shape()[-1]],initializer=tf.ones_initializer())
    beta = tf.get_variable(name='beta',shape=[inputs.get_shape()[-1]],initializer=tf.zeros_initializer())
    pop_mean = tf.get_variable(name='moving_mean',shape=[inputs.get_shape()[-1]],initializer=tf.zeros_initializer(),trainable=False)
    pop_var = tf.get_variable(name='moving_variance',shape=[inputs.get_shape()[-1]],initializer=tf.ones_initializer(),trainable=False)
    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,list(range(len(inputs.get_shape())-1)))
        train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
    else:

        return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta,scale,variance_epsilon=epsilon)
# %%
def BiGRU(layer_name,x,n_hidden,GRU_layer_num=3,input_dropout=0.5,output_dropout=0.5):
    #x = tf.transpose(x, [1, 0, 2])
    #x = tf.reshape(x, [-1, n_input])
    #x = tf.split(x, n_steps)
    n_steps = x.get_shape().as_list()[1]
    input_x = tf.unstack(x,num=n_steps,axis=1) #分解成nstep个矩阵
    stacked_fw_gru = []
    stacked_bw_gru = []
    for i in range(GRU_layer_num):
        # 正向
        gru_fw_cell = tf.contrib.rnn.GRUCell(n_hidden, kernel_initializer=tf.contrib.layers.xavier_initializer())
        #gru_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # dropout
        cell_fw_dr = tf.contrib.rnn.DropoutWrapper(gru_fw_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        stacked_fw_gru.append(cell_fw_dr)
        # 反向
        gru_bw_cell = tf.contrib.rnn.GRUCell(n_hidden, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # dropout
        cell_bw_dr = tf.contrib.rnn.DropoutWrapper(gru_bw_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        stacked_bw_gru.append(cell_bw_dr)

    with tf.name_scope(layer_name):
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(
            stacked_fw_gru,
            stacked_bw_gru,
            input_x,
            dtype=tf.float32
        )
    outputs = tf.transpose(outputs, [1, 0, 2])  # (n,n_step,n_hidden*2)
    return outputs
def BiLSTM(layer_name,x,n_hidden,GRU_layer_num=3,input_dropout=0.5,output_dropout=0.5):
    #x = tf.transpose(x, [1, 0, 2])
    #x = tf.reshape(x, [-1, n_input])
    #x = tf.split(x, n_steps)
    n_steps = x.get_shape().as_list()[1]
    input_x = tf.unstack(x,num=n_steps,axis=1) #分解成nstep个矩阵
    stacked_fw_gru = []
    stacked_bw_gru = []
    for i in range(GRU_layer_num):
        # 正向
        #gru_fw_cell = tf.contrib.rnn.GRUCell(n_hidden, kernel_initializer=tf.contrib.layers.xavier_initializer())
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        # dropout
        cell_fw_dr = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        stacked_fw_gru.append(cell_fw_dr)
        # 反向
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
        # dropout
        cell_bw_dr = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
        stacked_bw_gru.append(cell_bw_dr)

    with tf.name_scope(layer_name):
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(
            stacked_fw_gru,
            stacked_bw_gru,
            input_x,
            dtype=tf.float32
        )
    outputs = tf.transpose(outputs, [1, 0, 2])  # (n,n_step,n_hidden*2)
    return outputs
def LSTM2D(layer_name, x, n_hidden_units, n_outputs):
    # 对 weights biases 初始值的定义
    # X : batch,28 nsteps,28 inputs
    n_steps = int(x.get_shape().as_list()[1])
    print(n_steps)
    n_inputs = x.get_shape().as_list()[2]
    with tf.variable_scope(layer_name):
        weights = {
            # shape (28, 128)
            'in': tf.get_variable(name="weight_in", shape=[n_inputs, n_hidden_units],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            # shape (128, 10)
            'out': tf.get_variable(name='weight_out', shape=[n_hidden_units, n_outputs],
                                   initializer=tf.contrib.layers.xavier_initializer())
        }
        biases = {
            # shape (128, )
            'in': tf.get_variable(name="biases_in", shape=[n_hidden_units, ], initializer=tf.constant_initializer(0.1)),
            # shape (10, )
            'out': tf.get_variable(name="biases_out", shape=[n_outputs, ], initializer=tf.constant_initializer(0.1))
        }
    x = tf.transpose(x, [1, 0, 2])
    _X = tf.reshape(x, [-1, n_inputs])
    _H = tf.matmul(_X, weights['in']) + biases['in']
    _Hsplit = tf.split(value=_H, num_or_size_splits=n_steps,axis=0)
    with tf.variable_scope(layer_name) as scope:
        #scope.reuse_variables()
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.nn.static_rnn(lstm_cell, _Hsplit, dtype=tf.float32)
    _O = tf.matmul(_LSTM_O[-1], weights['out']) + biases['out']

    return {
        'X': _X, 'H': _H, 'Hsplit': _Hsplit,
        'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O
    }

def pool2D(layer_name, x, kernel_size=[1, 2, 2, 1], stride=[1, 2, 2, 1],padding='SAME',is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel_size, strides=stride, padding=padding, name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel_size, strides=stride, padding=padding, name=layer_name)
    return x
def pool1D(layer_name, x, window_shape=[1], stride=[2],padding='SAME',is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.pool(x,window_shape=window_shape,pooling_type='MAX',strides=stride,padding=padding,name = layer_name)
    else:
        x = tf.nn.pool(x,window_shape=window_shape,pooling_type='AVG',strides=stride,padding=padding,name = layer_name)
    return x

# %%
def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, list(range(len(x.get_shape()) - 1)))
    print (x.get_shape())
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


# %%
def FC_layer(layer_name, x, out_nodes,trainable,activation_fn=True,l2_value=0.00004):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            trainable=(trainable is not None),
                            regularizer=tf.contrib.layers.l2_regularizer(l2_value),
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            trainable=(trainable is not None),
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])  # flatten into 1D

        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if (activation_fn):
            x = tf.nn.relu(x)
        return x


# %%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    labels = tf.one_hot(labels,2)
    #print(labels)
    with tf.name_scope('loss') as scope:
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=labels, logits=logits,pos_weight=7.0, name='cross-entropy')
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross-entropy')
        #cross_entropy = tf.losses.mean_squared_error(logits, labels)
        #print(cross_entropy)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        #print(loss)
        #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #loss += tf.add_n(reg_variables)
        tf.summary.scalar(scope + '/loss', loss)
        return loss


# %%
def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor,
    """
    #labels = tf.reshape(labels, [-1, 1])
    with tf.name_scope('softmax') as scope:
        softmax = tf.nn.softmax(logits)
        prediction = tf.nn.top_k(softmax, 1)
        #max_probability = tf.reduce_max(probability,axis=1)
    with tf.name_scope('accuracy') as scope:
        #correct_pred = tf.equal(tf.arg_max(logits, 1),tf.arg_max(labels, 1))
        correct_pred = tf.equal(tf.arg_max(logits, 1),labels)
        correct_pred = tf.reshape(correct_pred, [-1])
        #accuracy_test = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100.0
        #accuracy = tf.reduce_mean(tf.cast(bool_list_correct_prediction, tf.float32))
        #tf.summary.scalar(scope + '/accuracy', accuracy)
    with tf.name_scope('auc') as scope:
        labels_split0,labels_split1 = tf.split(tf.one_hot(labels,2),num_or_size_splits=2,axis=1)
        softmax_split0, softmax_split1 = tf.split(softmax, num_or_size_splits=2, axis=1)
        auc,update_op = tf.metrics.auc(labels=labels_split1,predictions=softmax_split1)
    return prediction[0],correct_pred,softmax,auc,update_op#,softmax

def no_onehot_accuracy(Logits,y_input):
    one = tf.ones_like(Logits)
    zero = tf.zeros_like(Logits)
    Logits = tf.where(Logits <0.5, x=zero, y=one)
    batch_bool = tf.equal(Logits, y_input)
    batch_bool = tf.reshape(batch_bool, [-1])
    acc = tf.reduce_mean(tf.cast(bool_list_correct_prediction, tf.float32))
    return batch_bool#,acc
# %%
def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
        the number of correct predictions
    """
    correct = tf.equal(tf.arg_max(logits, 1),labels)
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct


# %%
def optimize(loss, learning_rate):#,global_step=0):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)#,global_step=global_step)
        return train_op


# %%
def load(data_path, session):
    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        with tf.variable_scope(key, reuse=True):
            for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                session.run(tf.get_variable(subkey).assign(data))


# %%
def test_load():
    data_path = './/vgg16_pretrain//vgg16.npy'

    data_dict = np.load(data_path, encoding='latin1').item()
    keys = sorted(data_dict.keys())
    for key in keys:
        weights = data_dict[key][0]
        biases = data_dict[key][1]
        print('\n')
        print(key)
        print('weights shape: ', weights.shape)
        print('biases shape: ', biases.shape)


# %%跳过加载哪几层的参数，VGG16最后一层的输出神经元有1000个，与我们的模型不匹配，所以要跳过加载VGG16最后一层的参数。
def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        #print (key)
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                if (len(data_dict[key]) >= 3):
                    #print(key)
                    session.run(tf.get_variable('moving_variance').assign(data_dict[key][0]))
                    session.run(tf.get_variable('moving_mean').assign(data_dict[key][1]))
                    session.run(tf.get_variable('beta').assign(data_dict[key][2]))
                elif(len(data_dict[key])>=2):
                    #print (key)
                    session.run(tf.get_variable('weights').assign(data_dict[key][0]))
                    session.run(tf.get_variable('biases').assign(data_dict[key][1]))
                    #for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    #    session.run(tf.get_variable(subkey).assign(data))
                else:
                    #print (key)
                    session.run(tf.get_variable("weights").assign(data_dict[key][0]))

# %%
def print_all_variables(train_only=True):
    """Print all trainable and non-trainable variables
    without tl.layers.initialize_global_variables(sess)

    Parameters
    ----------
    train_only : boolean
        If True, only print the trainable variables, otherwise, print all variables.
    """
    # tvar = tf.trainable_variables() if train_only else tf.all_variables()
    if train_only:
        t_vars = tf.trainable_variables()
        print("  [*] printing trainable variables")
    else:
        try:  # TF1.0
            t_vars = tf.global_variables()
        except:  # TF0.12
            t_vars = tf.all_variables()
        print("  [*] printing global variables")
    for idx, v in enumerate(t_vars):
        print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

    # %%


##***** the followings are just for test the tensor size at diferent layers *********##

# %%
def weight(kernel_shape, is_uniform=True):
    ''' weight initializer
    Args:
        shape: the shape of weight
        is_uniform: boolen type.
                if True: use uniform distribution initializer
                if False: use normal distribution initizalizer
    Returns:
        weight tensor
    '''
    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())
    return w


# %%
def bias(bias_shape):
    '''bias initializer
    '''
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b

# %% GoogleNet node_id_to_name 的映射函数
def map():
    uid_to_human = {}
    for line in tf.gfile.GFile('imagenet_synset_to_human_label_map.txt').readlines():
        items = line.strip().split('\t')
        uid_to_human[items[0]] = items[1]

    node_id_to_uid = {}
    for line in tf.gfile.GFile('imagenet_2012_challenge_label_map_proto.pbtxt').readlines():
        if line.startswith('  target_class:'):
            target_class = int(line.split(': ')[1])
        if line.startswith('  target_class_string:'):
            target_class_string = line.split(': ')[1].strip('\n').strip('\"')
            node_id_to_uid[target_class] = target_class_string

    node_id_to_name = {}
    for key, value in node_id_to_uid.items():
        node_id_to_name[key] = uid_to_human[value]
    return node_id_to_name

def embedding(layer_name,x,vocabulary_size, embedding_size):
    with tf.variable_scope(layer_name):
        word_embeddings = tf.get_variable("word_embeddings", [vocabulary_size, embedding_size])
        embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, x)
        #embedded_word_ids = tf.contrib.layers.embed_sequence(x, vocab_size=22, embed_dim=50)
    return embedded_word_ids

def parse_function(example_proto):
    # 只接受一个输入：example_proto，也就是序列化后的样本tf_serialized
    dics = {  # 这里没用default_value，随后的都是None
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

        # 使用 VarLenFeature来解析
        'Amino': tf.FixedLenFeature(shape=(800), dtype=tf.int64, default_value=None)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    # 解码字符

    return parsed_example

def parse_function_1(example_proto):
    # 只接受一个输入：example_proto，也就是序列化后的样本tf_serialized
    dics = {  # 这里没用default_value，随后的都是None
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),

        # 使用 VarLenFeature来解析
        'Amino_acids': tf.FixedLenFeature(shape=(), dtype=tf.string, default_value=None),
        'Amino_acids_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64, default_value=None)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    parsed_example['Amino_acids'] = tf.decode_raw(parsed_example['Amino_acids'], np.float32)
    parsed_example['Amino_acids'] = tf.reshape(parsed_example['Amino_acids'],[20,100])
    # 解码字符

    return parsed_example

def parse_group_train_2020(example_proto):
    dics = {  # 这里没用default_value，随后的都是None
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
        'data': tf.FixedLenFeature(shape=(800,), dtype=tf.float32, default_value=None)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    parsed_example['data1'],parsed_example['data2'] = tf.split(parsed_example['data'], [400,400], 0)
    parsed_example['data1'] = tf.reshape(parsed_example['data1'], [20,20])
    parsed_example['data2'] = tf.reshape(parsed_example['data2'], [20,20])

    return parsed_example


def parse_group_test_2020(example_proto):
    dics = {  # 这里没用default_value，随后的都是None
        'label': tf.FixedLenFeature(shape=(), dtype=tf.int64, default_value=None),
        'data': tf.FixedLenFeature(shape=(802,), dtype=tf.float32, default_value=None)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    parsed_example['data1'], parsed_example['data2'],parsed_example['extra_info'] = tf.split(parsed_example['data'], [400, 400,2], 0)
    parsed_example['data1'] = tf.reshape(parsed_example['data1'], [20, 20])
    parsed_example['data2'] = tf.reshape(parsed_example['data2'], [20, 20])
    parsed_example['extra_info'] = tf.reshape(parsed_example['extra_info'],[2,])

    return parsed_example

def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    print(static_tensor_shape)
    dynamic_tensor_shape = tf.shape(tensor)
    print(dynamic_tensor_shape)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        print(index,dim)
        if dim is not None:
            combined_shape.append(dim)
        else:
            print("dynamic_tensor_shape[index],",dynamic_tensor_shape[index])
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def Squeeze_excitation_layer(feature_map,index, ratio=2):
    with tf.variable_scope("se_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        channel_avg_weights = tf.layers.average_pooling1d(
            feature_map,
            feature_map_shape[1],
            1,
            padding='valid'
        )
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[2]])
        fc_1 = tf.layers.dense(
            inputs=channel_avg_reshape,
            units=feature_map_shape[2]/ 2,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[2],
            name="fc_2",
            activation=tf.nn.sigmoid
        )
        excitation = tf.reshape(fc_2, [-1, 1, feature_map_shape[2]])
        scale = feature_map * excitation

        return scale

def convolutional_block_attention_module_2D(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention
        channel_avg_weights = tf.nn.avg_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_max_weights = tf.nn.max_pool(
            value=feature_map,
            ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID'
        )
        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0], 1, feature_map_shape[3]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[3] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[3],
            name="fc_2",
            activation=None
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
        print(feature_map.shape.as_list())
        print(channel_attention.shape.as_list())
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        print(feature_map_with_channel_attention.shape.as_list())
        # spatial attention
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                     1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
        spatial_attention = tf.layers.conv2d(
            channel_wise_pooling,
            1,
            [7, 7],
            padding='same',
            activation=tf.nn.sigmoid,
            name="spatial_attention_conv"
        )
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention
def convolutional_block_attention_module_1D(feature_map, index, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    with tf.variable_scope("cbam_%s" % (index)):
        feature_map_shape = combined_static_and_dynamic_shape(feature_map)
        # channel attention
        channel_avg_weights = tf.layers.average_pooling1d(
            feature_map,
            feature_map_shape[1],
            1,
            padding='valid'
        )
        channel_max_weights = tf.layers.max_pooling1d(
            feature_map,
            feature_map_shape[1],
            1,
            padding='valid'
        )

        channel_avg_reshape = tf.reshape(channel_avg_weights,
                                         [feature_map_shape[0],1,feature_map_shape[2]])
        channel_max_reshape = tf.reshape(channel_max_weights,
                                         [feature_map_shape[0],1,feature_map_shape[2]])
        channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

        fc_1 = tf.layers.dense(
            inputs=channel_w_reshape,
            units=feature_map_shape[2] * inner_units_ratio,
            name="fc_1",
            activation=tf.nn.relu
        )
        fc_2 = tf.layers.dense(
            inputs=fc_1,
            units=feature_map_shape[2],
            name="fc_2",
            activation=None
        )
        channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
        channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
        channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, feature_map_shape[2]])
        print(feature_map.shape.as_list())
        print(channel_attention.shape.as_list())
        feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
        print(feature_map_with_channel_attention.shape.as_list())
        # spatial attention
        channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=2)
        channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=2)

        channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1],1])
        channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                              shape=[feature_map_shape[0], feature_map_shape[1],1])

        channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=2)
        spatial_attention = tf.layers.conv1d(
            channel_wise_pooling,
            1,
            7,
            padding='same',
            activation=tf.nn.relu,
            name="spatial_attention_conv"
        )
        feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
        return feature_map_with_attention