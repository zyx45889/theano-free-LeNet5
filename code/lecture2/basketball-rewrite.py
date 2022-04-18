from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import math
import matplotlib.pyplot as plt

def load_data_frame2():
    hog_file = open('../../selected/hog.pkl', 'rb')
    data_xx = pickle.load(hog_file)
    label_file = open('../../selected/label.pkl', 'rb')
    data_yy = pickle.load(label_file)
    # print(len(data_x[0]))
    data_x=[]
    data_y=[]
    l=len(data_xx)
    for i in range(l-1):
        vector=[]
        for j in range(len(data_xx[0])):
            vector.append(data_xx[i][j])
        for j in range(len(data_xx[0])):
            vector.append(data_xx[i+1][j])
        if (data_yy[i])&(data_yy[i+1]):
            data_y.append(1)
        else:
            data_y.append(0)
        data_x.append(vector)
        i+=1
    l=len(data_x)
    train_set, valid_set, test_set = data_x[0:int(l*0.6)-1],data_x[int(l*0.6):int(l*0.8)-1],data_x[int(l*0.8):l-1]
    train_label, valid_label, test_label = data_y[0:int(l * 0.6) - 1],data_y[int(l * 0.6): int(l * 0.8) - 1], data_y[int(l * 0.8): l - 1]

    test_set_x = numpy.asarray(test_set)
    valid_set_x = numpy.asarray(valid_set)
    train_set_x = numpy.asarray(train_set)
    test_set_y = numpy.asarray(test_label)
    valid_set_y = numpy.asarray(valid_label)
    train_set_y = numpy.asarray(train_label)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_data():
    hog_file = open('../../selected/hog.pkl', 'rb')
    data_x = pickle.load(hog_file)
    label_file = open('../../selected/label.pkl', 'rb')
    data_y = pickle.load(label_file)
    # print(len(data_x[0]))
    l=len(data_x)
    train_set, valid_set, test_set = data_x[0:int(l*0.6)-1],data_x[int(l*0.6):int(l*0.8)-1],data_x[int(l*0.8):l-1]
    train_label, valid_label, test_label = data_y[0:int(l * 0.6) - 1],data_y[int(l * 0.6): int(l * 0.8) - 1], data_y[int(l * 0.8): l - 1]

    test_set_x = numpy.asarray(test_set)
    valid_set_x = numpy.asarray(valid_set)
    train_set_x = numpy.asarray(train_set)
    test_set_y = numpy.asarray(test_label)
    valid_set_y = numpy.asarray(valid_label)
    train_set_y = numpy.asarray(train_label)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

learning_rate=0.03
n_epochs=30
batch_size=20
datasets = load_data()
n_in = 1296
n_out = 2

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.shape[0] // batch_size
n_valid_batches = valid_set_x.shape[0] // batch_size
n_test_batches = test_set_x.shape[0] // batch_size

W = numpy.zeros((n_in, n_out), dtype='float64')
b = numpy.zeros((n_out,), dtype='float64')
p_y_given_x = numpy.zeros((batch_size, n_out), dtype='float64')
g_W = numpy.zeros((n_in, n_out), dtype='float64')
g_b = numpy.zeros((n_out,), dtype='float64')

###############
# TRAIN MODEL #
###############
print('... training the model')
patience = 20
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience // 2)
pred=[]
best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()
lamda=0.01
done_looping = False
linear_search=0
epoch = 0
print(n_train_batches,validation_frequency)
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        index = minibatch_index
        x = train_set_x[index * batch_size: (index + 1) * batch_size]
        y = train_set_y[index * batch_size: (index + 1) * batch_size]
        # 计算分布函数
        thi = numpy.matmul(x, W)
        for i in range(batch_size):
            p_y_given_x[i] = thi[i] + b
        p_y_given_x = numpy.exp(p_y_given_x)
        sum = numpy.sum(p_y_given_x, axis=1)
        for i in range(batch_size):
            p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
        y_pred = numpy.argmax(p_y_given_x, axis=1)

        # 计算梯度
        y_is_j = []
        for i in range(batch_size):
            nowyisj = []
            for j in range(n_out):
                if y[i]==j:
                    nowyisj.append(1)
                else:
                    nowyisj.append(0)
            y_is_j.append(nowyisj)
        umm = numpy.asarray(y_is_j)
        coef=umm-p_y_given_x
        for a in range(batch_size):
            for i in range(n_in):
                for j in range(n_out):
                    g_W[i][j]-=x[a][i]*coef[a][j]
        g_W/=batch_size
        g_W+=lamda*W
        g_b=-1.0*numpy.mean(coef,axis=0)+lamda*b
        # for i in range(batch_size):
        #     for a in range(n_in):
        #         for b in range(n_out):
        #             g_W[a][b] += p_y_given_x[i][b] * x[i][a]
        #             if (b == y[i]):
        #                 g_W[a][b] -= x[i][a]
        # g_W+=lamda*W
        # g_W = numpy.divide(g_W, batch_size)
        # for i in range(batch_size):
        #     for j in range(n_out):
        #         g_b[j] += p_y_given_x[i][j]
        #         if j == y[i]:
        #             g_b[j] -= 1
        # g_b = numpy.divide(g_b, batch_size)
        # g_b+=lamda*b
        if linear_search==0:
            W = W - learning_rate * g_W
            b = b - learning_rate * g_b
        else:
            a=1.0
            c=0.5
            phy=0.5
            m=numpy.sum(numpy.square(g_W))
            fx=0.0
            for i in range(batch_size):
                fx-=math.log(p_y_given_x[i][y[i]])
            fx/=batch_size
            # print(fx, ":")
            while(1):
                # print(a)
                thi = numpy.matmul(x, W-a*g_W)
                for i in range(batch_size):
                    p_y_given_x[i] = thi[i] + b
                p_y_given_x = numpy.exp(p_y_given_x)
                sum = numpy.sum(p_y_given_x, axis=1)
                for i in range(batch_size):
                    p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
                nowfx = 0.0
                for i in range(batch_size):
                    nowfx -= math.log(p_y_given_x[i][y[i]])
                nowfx /= batch_size
                # print(nowfx)
                if nowfx<=fx-a*c*m:
                    break
                else:
                    a=a*phy
            W = W - a * g_W
            # print(a)
            a = 1.0
            c = 0.5
            phy = 0.5
            m = numpy.sum(numpy.square(g_b))
            while (1):
                # print(a)
                thi = numpy.matmul(x, W )
                for i in range(batch_size):
                    p_y_given_x[i] = thi[i] + b-a*g_b
                p_y_given_x = numpy.exp(p_y_given_x)
                sum = numpy.sum(p_y_given_x, axis=1)
                for i in range(batch_size):
                    p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
                nowfx = 0.0
                for i in range(batch_size):
                    nowfx -= math.log(p_y_given_x[i][y[i]])
                nowfx /= batch_size
                if nowfx <= fx - a * c * m:
                    break
                else:
                    a = a * phy
            b=b-a*g_b

        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter+1) % validation_frequency == 0:
            # compute zero-one loss on validation set
            validation_losses = 0
            for index in range(n_valid_batches):
                x = valid_set_x[index * batch_size: (index + 1) * batch_size]
                y = valid_set_y[index * batch_size: (index + 1) * batch_size]
                # 计算分布函数
                thi = numpy.matmul(x, W)
                for i in range(batch_size):
                    p_y_given_x[i] = thi[i] + b
                p_y_given_x = numpy.exp(p_y_given_x)
                sum = numpy.sum(p_y_given_x, axis=1)
                for i in range(batch_size):
                    p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
                y_pred = numpy.argmax(p_y_given_x, axis=1)
                thi = numpy.array(y_pred - y)
                validation_losses += len(numpy.argwhere(thi.astype(int))) / len(y)
            this_validation_loss = validation_losses/n_valid_batches


            print(
                'epoch %i, minibatch %i/%i, validation error %f %%' %
                (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                )
            )

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                   improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = this_validation_loss
                # test it on the test set
                test_losses = 0
                pred.clear()
                for index in range(n_test_batches):
                    x = test_set_x[index * batch_size: (index + 1) * batch_size]
                    y = test_set_y[index * batch_size: (index + 1) * batch_size]
                    # 计算分布函数
                    thi = numpy.matmul(x, W)
                    for i in range(batch_size):
                        p_y_given_x[i] = thi[i] + b
                    p_y_given_x = numpy.exp(p_y_given_x)
                    sum = numpy.sum(p_y_given_x, axis=1)
                    for i in range(batch_size):
                        p_y_given_x[i] = numpy.divide(p_y_given_x[i], sum[i])
                        pred.append(p_y_given_x[i][1])
                    y_pred = numpy.argmax(p_y_given_x, axis=1)
                    thi = numpy.array(y_pred - y)
                    test_losses += len(numpy.argwhere(thi.astype(int))) / len(y)
                test_score = test_losses/n_test_batches

                print(
                    (
                        '     epoch %i, minibatch %i/%i, test error of'
                        ' best model %f %%'
                    ) %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        test_score * 100.
                    )
                )

end_time = timeit.default_timer()
print(
    (
        'Optimization complete with best validation score of %f %%,'
        'with test performance %f %%'
    )
    % (best_validation_loss * 100., test_score * 100.)
)
print('The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time)))
print(('The code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

n = 100
x = []
y = []
for i in range(n):
    threshold = (max(pred) - min(pred)) / n * i
    tt = 0
    tf = 0
    pos=0
    neg=0
    for j in range(len(pred)):
        if pred[j] > threshold:
            if test_set_y[j] == 1:
                tt += 1
            else:
                tf += 1
        if test_set_y[j]==1:
            pos+=1
        else:
            neg+=1
    y.append(tt / pos)
    x.append(tf / neg)
plt.plot(x, y, 'ro-')
plt.xlabel('false alarm number')
plt.ylabel('missed positive number')
plt.title('ROC Curve')
plt.show()