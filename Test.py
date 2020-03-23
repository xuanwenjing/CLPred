import tensorflow as tf
import numpy as np
import re
import os
import math
from Bio import SeqIO
import sklearn.metrics as metrics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

dir = './model'
test_file = './DATA/DC_final/DC_final.fasta'
def input_fasta_800(filePath):
    X_train = []
    Y_train = []
    fasta_seq = SeqIO.parse(filePath,'fasta')
    for fasta in fasta_seq:
        id,seq = fasta.id,str(fasta.seq)
        X_train.append(seq)
        label = str.split(id,'id')
        if label[0] == '1':
            Y_train.append(1)
        else:
            Y_train.append(0)

    for i in range(0, len(X_train)):
        train = []
        st = str(X_train[i])
        trainStr = ''
        for ch in st:
            if (ch == 'B' or ch == 'J' or ch == 'O' or ch == 'U' or ch == 'Z'):
                trainStr += 'X'
            else:
                trainStr += ch

        train.append(trainStr)
        X_train[i] = train

    amino_acids = 'ACDEFGHIKLMNPQRSTVWXY'
    for i in range(0, len(X_train)):
        train = []
        st = str(X_train[i])
        trainStr = ''
        for ch in st:
            if (ch in amino_acids):
                trainStr += ch
        train.append(trainStr)
        X_train[i] = train

    X = []
    Y = []
    for i in range(len(X_train)):
        if ( len(X_train[i][0]) <= 800):
            X.append(X_train[i][0])
            Y.append(Y_train[i])

    X_train = np.array(X)
    y_train = np.array(Y)

    X_train = X_train.reshape(len(X_train), 1)

    amino_acids = 'ACDEFGHIKLMNPQRSTVWXY'
    embed = []
    for i in range(0, len(X_train)):
        length = len(X_train[i][0])
        pos = []
        counter = 0
        st = X_train[i][0]
        for c in st:
            AMINO_INDEX = amino_acids.index(c)
            pos.append(AMINO_INDEX+1)
            counter += 1
        while (counter < 800):
            pos.append(0)
            counter += 1
        embed.append(pos)
    embed = np.array(embed)

    X_train = np.array(embed,dtype=np.int32)
    y_train = np.array(Y_train,dtype=np.int32)
    return X_train,y_train
X_test_800,Y_test_800 = input_fasta_800(test_file)
result = np.zeros([len(X_test_800),11])
for count in range(1,10):
    final_pred = np.zeros(len(Y_test_800))
    final_class = np.zeros(len(Y_test_800))
    with tf.Session() as sess:
        print('model '+str(count)+' is precessing....')
        ckpt = tf.train.get_checkpoint_state(dir + '/'+str(count)+'/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()
        input = graph.get_tensor_by_name("input_x:0")
        op_to_restore = graph.get_tensor_by_name("FC3/BiasAdd:0")
        logits = sess.run(op_to_restore, feed_dict={input: X_test_800})
        prediction = sess.run(tf.reshape(tf.nn.softmax(logits)[:,1], [-1]))
        for i in range(len(result)):
            result[i][count-1] = prediction[i]
    tf.reset_default_graph()
final_class = []
final_pred = []
with tf.Session() as sess:
    for i in range(len(result)):
        result[i][10] = np.mean(result[i][0:10])
    for i in range(len(result)):
        final_pred.append(result[i][10])
        if result[i][10]>=0.5:
            final_class.append(1)
        else:
            final_class.append(0)
    confusion_matrix = metrics.confusion_matrix(y_true=Y_test_800, y_pred=final_class)
    print(confusion_matrix)
    fpr, tpr, threshold = metrics.roc_curve(Y_test_800, final_pred)  ###计算真正率和假正率
    AUC = metrics.auc(fpr, tpr)
    try:
        TP = confusion_matrix[1][1]
    except:
        TP = 0
    try:
        TN = confusion_matrix[0][0]
    except:
        TN = 0
    try:
        FP = confusion_matrix[0][1]
    except:
        FP = 0
    try:
        FN = confusion_matrix[1][0]
    except:
        FN = 0
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F_score = 2 * Recall * Precision / (Recall + Precision)
    NPV = TN / (TN + FN)
    print("TP:", TP, "TN",TN, "FP",FP, "FN",FN)
    print("AUC:", AUC)
    print("ACC:", ACC)
    print("MCC:", MCC)
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F_score:", F_score)
    print("NPV:", NPV)