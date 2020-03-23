import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import input_data
import tools
import os
from tensorflow.contrib import rnn
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import shuffle

train_file = './DATA/Train/train_csv.fasta'
train_logdir = './model/'
from imblearn.under_sampling import RandomUnderSampler

with tf.device('/cpu:0'):
    # function to get and process images or data.
    X_train_1, Y_train_1 = input_data.input_fasta_800(train_file)
vocabulary_size = 800
embedding_size = 50
BATCH_SIZE = 128

x_input = tf.placeholder(tf.int32, shape=[None, 800], name='input_x')
y_input = tf.placeholder(tf.int64, shape=None, name='input_y')
batch_size = tf.placeholder(tf.int32, [])

x = tools.embedding('Embedding', x_input, vocabulary_size, embedding_size)  # none,800,50
print(x.get_shape().as_list())
x = tf.reshape(x, [-1, 800, 50])

x1 = tools.conv1D("Cov1_1", x, out_channels=64, kernel_size=2, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
print(x1.get_shape().as_list())
x1 = tools.pool1D('Cov1_1_pool', x1, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x2 = tools.conv1D("Cov1_2", x, out_channels=64, kernel_size=3, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x2 = tools.pool1D('Cov1_2_pool', x2, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x3 = tools.conv1D("Cov1_3", x, out_channels=64, kernel_size=8, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x3 = tools.pool1D('Cov1_3_pool', x3, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x4 = tools.conv1D("Cov1_4", x, out_channels=64, kernel_size=9, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x4 = tools.pool1D('Cov1_4_pool', x4, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x5 = tools.conv1D("Cov1_5", x, out_channels=64, kernel_size=4, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x5 = tools.pool1D('Cov1_5_pool', x5, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x6 = tools.conv1D("Cov1_6", x, out_channels=64, kernel_size=5, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x6 = tools.pool1D('Cov1_6_pool', x6, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x7 = tools.conv1D("Cov1_7", x, out_channels=64, kernel_size=6, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x7 = tools.pool1D('Cov1_7_pool', x7, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x8 = tools.conv1D("Cov1_8", x, out_channels=64, kernel_size=7, stride=1, padding='SAME', activation_fn=True,
                  trainable=True, l2_value=0.0001)
x8 = tools.pool1D('Cov1_8_pool', x8, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)

x = tf.concat([x1, x2, x3, x4, x5, x6, x7, x8], axis=2, name='concat1')
print(x.get_shape().as_list())
x = tools.pool1D('Cov1_pool', x, window_shape=[2], stride=[2], padding='VALID', is_max_pool=False)

x9 = tools.conv1D("Cov2_1", x, out_channels=64, kernel_size=5, stride=1, padding='SAME',
                  activation_fn=True, trainable=True, l2_value=0.0001)
x9 = tools.pool1D('Cov2_1_pool', x9, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x10 = tools.conv1D("Cov2_2", x, out_channels=64, kernel_size=9, stride=1, padding='SAME',
                   activation_fn=True, trainable=True, l2_value=0.0001)
x10 = tools.pool1D('Cov2_2_pool', x10, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x11 = tools.conv1D("Cov2_3", x, out_channels=64, kernel_size=13, stride=1, padding='SAME',
                   activation_fn=True, trainable=True, l2_value=0.0001)
x11 = tools.pool1D('Cov2_3_pool', x11, window_shape=[5], stride=[1], padding='SAME', is_max_pool=False)
x = tf.concat([x9, x10, x11], axis=2, name='concat2')
x = tools.pool1D('Cov2_pool', x, window_shape=[2], stride=[2], padding='VALID', is_max_pool=False)
print(x.get_shape().as_list())
x = tools.BiLSTM('birnn', x, 128, GRU_layer_num=3)
print('BIRNN', x.get_shape().as_list())

x = tf.layers.flatten(x, 'flatten')
x = tools.FC_layer('FC2', x, 256, True, activation_fn=True, l2_value=0.001)
Logits = tools.FC_layer('FC3', x, 2, True, activation_fn=False, l2_value=0.001)
loss = tools.loss(Logits, y_input)

op = tools.optimize(loss, learning_rate=0.0001)
prediction, batch_bool, pred, auc, update_op = tools.accuracy(Logits, y_input)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
with tf.Session() as sess:
    X_train_1, Y_train_1 = shuffle(X_train_1, Y_train_1)
    kf = KFold(n_splits=10)
    count = 1
    for train_index, test_index in kf.split(X_train_1):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        X_train_, Y_train_ = X_train_1[train_index], Y_train_1[train_index]
        X_val, Y_val = X_train_1[test_index], Y_train_1[test_index]
        X_val, Y_val = RandomUnderSampler().fit_sample(X_val, Y_val)
        test_acc_list = []
        max_acc = 0
        for epoch in range(70):
            tra_loss = 0
            all_correct_prediction_train = []
            all_correct_prediction_test = []
            tra_acc = 0
            test_acc = 0
            AUC = 0
            X_train, Y_train = RandomUnderSampler().fit_sample(X_train_, Y_train_)
            X_train, Y_train = shuffle(X_train, Y_train)
            n_train = int(len(X_train) / BATCH_SIZE)
            for i in range(n_train):
                _, tra_loss, correct_list = sess.run([op, loss, batch_bool], feed_dict={
                    x_input: X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                    y_input: Y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                    batch_size: BATCH_SIZE,
                })
                all_correct_prediction_train = all_correct_prediction_train + list(correct_list)
            tra_acc = sess.run(tf.reduce_mean(tf.cast(all_correct_prediction_train, tf.float32)))
            print('epoch: %d, loss: %f, accuracy: %f%%,' % (epoch, tra_loss, tra_acc))

            if epoch % 1 == 0:
                # val
                all_correct_prediction_train = []
                n_val = int(len(X_val) / BATCH_SIZE)
                n_val_yushu = int(len(X_val % BATCH_SIZE)) #
                val_loss = 0
                for i in range(n_val):
                    val_loss, correct_list = sess.run([loss, batch_bool], feed_dict={
                        x_input: X_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                        y_input: Y_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                        batch_size: BATCH_SIZE,
                    })
                    all_correct_prediction_train = all_correct_prediction_train + list(correct_list)
                if (n_val_yushu != 0):
                    correct_list = sess.run(batch_bool, feed_dict={
                        x_input: X_val[n_val * BATCH_SIZE:n_val * BATCH_SIZE + n_val_yushu],
                        y_input: Y_val[n_val * BATCH_SIZE:n_val * BATCH_SIZE + n_val_yushu],
                        batch_size: n_val_yushu,
                    })
                    all_correct_prediction_train = all_correct_prediction_train + list(correct_list)
                val_acc = sess.run(tf.reduce_mean(tf.cast(all_correct_prediction_train, tf.float32)))
                val_acc_list.append(val_acc)
                print('**  Step %d, val loss = %f, val accuracy = %f%%  **' % (epoch, val_loss, val_acc))

            if epoch % 1 == 0:
                current_max = max(val_acc_list)
                path = train_logdir + str(counter) + '/' + str(count)
                if (current_max > max_acc):
                    if (os.path.exists(path)):
                        checkpoint_path = os.path.join(path, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch)
                        max_acc = current_max
                    else:
                        os.makedirs(path)
                        checkpoint_path = os.path.join(path, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=epoch)
                        max_acc = current_max
        count = count + 1