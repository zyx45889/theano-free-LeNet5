from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = numpy.asarray(data_x,dtype='float64')
        shared_y = numpy.asarray(data_y,dtype='float64')
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y.astype(int)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


learning_rate=0.13
n_epochs=1000
dataset='mnist.pkl.gz'
batch_size=60
datasets = load_data(dataset)
n_in = 784
n_out = 10

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

# with open("C:/Users/86133/Desktop/test.txt", "r") as f:
#     data = f.read()
# data=data.split()
# for i in range(784):
#     for j in range(10):
#         W[i][j]=data[i*10+j]
#
# for i in range(10):
#     b[i]=data[784*10-1+i]

###############
# TRAIN MODEL #
###############
print('... training the model')
# early-stopping parameters
patience = 100  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                              # found
improvement_threshold = 0.995  # a relative improvement of this much is
                              # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                              # go through this many
                              # minibatche before checking the network
                              # on the validation set; in this case we
                              # check every epoch

best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
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
        # for i in range(batch_size):
        #     W=W-learning_rate*numpy.matmul(numpy.reshape(x[i],(784,1)),numpy.reshape(y[i]-p_y_given_x[i],(1,10)))
        for i in range(batch_size):
            for a in range(n_in):
                for b in range(n_out):
                    g_W[a][b] += p_y_given_x[i][b] * x[i][a]
                    if (b == y[i]):
                        g_W[a][b] -= x[i][a]
        g_W = numpy.divide(g_W, batch_size)
        for i in range(batch_size):
            for j in range(n_out):
                g_b[j] += p_y_given_x[i][j]
                if j == y[i]:
                    g_b[j] -= 1
        g_b = numpy.divide(g_b, batch_size)
        W = W - learning_rate * g_W
        b = b - learning_rate * g_b

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

        if patience <= iter:
            done_looping = True
            break

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