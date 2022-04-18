from __future__ import print_function

__docformat__ = 'restructedtext en'


import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import matplotlib.pyplot as plt


def load_data():
    hog_file = open('../../selected/hog.pkl', 'rb')
    data_x = pickle.load(hog_file)
    label_file = open('../../selected/label.pkl', 'rb')
    data_y = pickle.load(label_file)
    # print(len(data_x[0]))
    l=len(data_x)
    # for i in range(l):
    #     data_x[i].extend(data_x[(i+1)%l])
    #     if (data_y[i]==1)&(data_y[(i+1)%l]==1):
    #         data_y[i]=1
    #     else:
    #         data_y[i]=0
    train_set, valid_set, test_set = data_x[0:int(l*0.6)-1],data_x[int(l*0.6):int(l*0.8)-1],data_x[int(l*0.8):l-1]
    train_label, valid_label, test_label = data_y[0:int(l * 0.6) - 1],data_y[int(l * 0.6): int(l * 0.8) - 1], data_y[int(l * 0.8): l - 1]
    # print(sum(test_label)/len(test_label),sum(valid_label)/len(valid_label))
    # def shared_datasetx(data, borrow=True):
    #     shared= theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=borrow)
    #     return shared
    # def shared_datasety(data, borrow=True):
    #     shared= theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=borrow)
    #     return T.cast(shared, 'int32')
    #
    test_set_x = numpy.asarray(test_set)
    valid_set_x = numpy.asarray(valid_set)
    train_set_x = numpy.asarray(train_set)
    test_set_y = numpy.asarray(test_label)
    valid_set_y = numpy.asarray(valid_label)
    train_set_y = numpy.asarray(train_label)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class hidden_layer(object):
    def __init__(self,n_in,n_out,batch,rng):
        self.W = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype='float64')
        self.b = numpy.zeros((n_out,), dtype='float64')
        self.z = numpy.zeros((batch,n_out),dtype='float64')
        self.a = numpy.zeros((batch,n_out),dtype='float64')
        self.delta = numpy.zeros((batch,n_out),dtype='float64')
        self.batch=batch

    def forward_compute_z_a(self,x):
        thi=numpy.matmul(x,self.W)
        for i in range(self.batch):
            self.z[i] = thi[i] + self.b
            self.a[i]=(numpy.exp(self.z[i])-numpy.exp(-self.z[i]))/(numpy.exp(self.z[i])+numpy.exp(-self.z[i]))
        return self.a

    def back_dalta(self,next_W,next_delta):
        tt=numpy.dot(next_delta,next_W.transpose())
        self.delta=tt*(1-self.a**2)

    def back_update_W_b(self,x,learning_rate,L2_reg):
        delta_W = -1.0*numpy.dot(x.transpose(),self.delta)/x.shape[0]
        delta_b = -1.0*numpy.mean(self.delta,axis=0)

        self.W-=learning_rate*(L2_reg*self.W+delta_W)
        self.b-=learning_rate*delta_b

class output_layer(object):
    def __init__(self,n_in,n_out,batch,rng):
        self.p_y_given_x = numpy.zeros((batch,n_out),dtype='float64')
        self.exp_x_multiply_W_plus_b=numpy.zeros((batch,n_out),dtype='float64')
        self.delta=numpy.zeros((batch,n_out),dtype='float64')
        self.W = numpy.zeros((n_in, n_out), dtype='float64')
        self.b = numpy.zeros((n_out,), dtype='float64')
        self.n_out=n_out
        self.batch=batch

    def forward_compute_p_y_given_x(self,x):
        thi=numpy.matmul(x,self.W)
        for i in range(self.batch):
            self.exp_x_multiply_W_plus_b[i] = numpy.exp(thi[i] + self.b)
        sigma=numpy.sum(self.exp_x_multiply_W_plus_b,axis=1)
        self.p_y_given_x=self.exp_x_multiply_W_plus_b/sigma.reshape(sigma.shape[0],1)

    def back_compute_delta(self,y):
        yy=numpy.zeros((y.shape[0],self.n_out))
        yy[numpy.arange(y.shape[0]),y]=1.0
        self.delta=yy-self.p_y_given_x

    def back_update_W_b(self,x,learning_rate,L2_reg):
        delta_W = -1.0 * numpy.dot(x.transpose(), self.delta) / x.shape[0]
        delta_b = -1.0 * numpy.mean(self.delta, axis=0)

        self.W -= learning_rate * (L2_reg * self.W + delta_W)
        self.b -= learning_rate * delta_b

class MLP(object):
    def __init__(self,batch,rng,n_in=1296,n_list_hidden_nodes=[10,4],n_out=2):
        self.hidden_layer_list=[]
        for i in range(len(n_list_hidden_nodes)):
            now_in=0
            if i==0:
                now_in=n_in
            else:
                now_in=n_list_hidden_nodes[i-1]
            now_out=n_list_hidden_nodes[i]
            self.hidden_layer_list.append(hidden_layer(now_in,now_out,batch=batch,rng=rng))
        self.output_layer=output_layer(n_list_hidden_nodes[len(n_list_hidden_nodes)-1],n_out,batch=batch,rng=rng)

    def feedforward(self,x):
        xx=x
        for each in self.hidden_layer_list:
            each.forward_compute_z_a(xx)
            xx=each.a
        self.output_layer.forward_compute_p_y_given_x(xx)

    def backpropagation(self,x,y,learning_rate,L2_reg):
        self.output_layer.back_compute_delta(y)
        xx=self.hidden_layer_list[-1].a
        self.output_layer.back_update_W_b(xx,learning_rate,L2_reg)
        next_W=self.output_layer.W
        next_delta=self.output_layer.delta
        i= len(self.hidden_layer_list)
        while i>0:
            curr_hidden_lay=self.hidden_layer_list[i-1]
            curr_hidden_lay.back_dalta(next_W,next_delta)

            if i>1:
                xx=self.hidden_layer_list[i-2].a
            else:
                xx=x
            curr_hidden_lay.back_update_W_b(xx,learning_rate,L2_reg)

            next_W=curr_hidden_lay.W
            next_delta=curr_hidden_lay.delta
            i-=1

def cal_loss(y_p,y):
    f=0
    y_cal=numpy.argmax(y_p,axis=1)
    # print(y_p,y)
    for i in range(y_cal.shape[0]):
        if y[i]!=y_cal[i]:
            f+=1
    return f/y_cal.shape[0]

def test_mlp(learning_rate=0.03, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    xx = []
    yy = []
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    rng = numpy.random.RandomState(1234)
    classifier = MLP(batch=batch_size,rng=rng)

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    patience = 500
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    print(validation_frequency)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            x=train_set_x[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
            y=train_set_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
            classifier.feedforward(x)
            classifier.backpropagation(x,y,learning_rate,L2_reg)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                this_validation_loss=0
                for index in range(n_valid_batches):
                    x=valid_set_x[index*batch_size:(index+1)*batch_size]
                    y=valid_set_y[index*batch_size:(index+1)*batch_size]
                    classifier.feedforward(x)
                    p_y_give_x = classifier.output_layer.p_y_given_x
                    this_validation_loss+=cal_loss(p_y_give_x,y)
                this_validation_loss/=n_valid_batches

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
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_score = 0
                    pred=[]
                    for index in range(n_test_batches):
                        x = test_set_x[index * batch_size:(index + 1) * batch_size]
                        y = test_set_y[index * batch_size:(index + 1) * batch_size]
                        classifier.feedforward(x)
                        p_y_give_x = classifier.output_layer.p_y_given_x
                        for i in range(batch_size):
                            pred.append(p_y_give_x[i][1])
                        test_score += cal_loss(p_y_give_x, y)
                    test_score /= n_valid_batches
                    n = 100
                    x = []
                    y = []
                    for i in range(n):
                        threshold = (max(pred) - min(pred)) / n * i
                        tt = 0
                        tf = 0
                        pos = 0
                        neg = 0
                        for j in range(len(pred)):
                            if pred[j] > threshold:
                                if test_set_y[j] == 1:
                                    tt += 1
                                else:
                                    tf += 1
                            if test_set_y[j] == 1:
                                pos += 1
                            else:
                                neg += 1
                        y.append(tt / pos)
                        x.append(tf / neg)
                    xx = x.copy()
                    yy = y.copy()
                    x.clear()
                    y.clear()
                    pred.clear()

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    plt.plot(xx, yy, 'ro-')
    plt.xlabel('false alarm number')
    plt.ylabel('missed positive number')
    plt.title('ROC Curve')
    plt.show()


if __name__ == '__main__':
    test_mlp()