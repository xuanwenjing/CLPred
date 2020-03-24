
import numpy as np

from Bio import SeqIO
from collections import Counter
def input_fasta_800(filePath):
    X_train = [] #['SFDFS','FSDFSD']
    Y_train = [] #[0,1,0,0,0,1]
    fasta_seq = SeqIO.parse(filePath,'fasta')
    for fasta in fasta_seq:
        id,seq = fasta.id,str(fasta.seq)
        X_train.append(seq)
        label = str.split(id, 'id')
        if label[0] == '1':
            Y_train.append(1)
        else:
            Y_train.append(0)
    each_len = np.zeros(12,np.int32)
    threshold = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200]
    for i in range(0, len(X_train)):
        train = []
        st = str(X_train[i])
        len_st = len(st)
        for t in range(len(threshold)-1):
            if ( threshold[t] <= len_st < threshold[t+1]):
                each_len[t]+=1
        trainStr = ''
        for ch in st:
            if (ch == 'B' or ch == 'J' or ch == 'O' or ch == 'U' or ch == 'Z'):
                trainStr += 'X'
            else:
                trainStr += ch

        train.append(trainStr)
        X_train[i] = train
    print(each_len)

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
    X_test = []
    Y_test = []
    for i in range(len(X_train)):
        if ( len(X_train[i][0]) <= 800):
            X.append(X_train[i][0])
            Y.append(Y_train[i])

    X_train = np.array(X)
    y_train = np.array(Y)

    X_train = X_train.reshape(len(X_train), 1)

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

    print(Counter(y_train))
    X_train = np.array(embed)
    y_train = np.array(y_train)
    return X_train,y_train



#input_fasta_group_train("train_csv.fasta")