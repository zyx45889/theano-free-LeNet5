from __future__ import print_function

__docformat__ = 'restructedtext en'


import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
from scipy import fftpack
from scipy import signal


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
        shared_x = numpy.zeros((len(data_x),28,28),dtype='float64')
        for i in range(len(data_x)):
            for j in range(28):
                for k in range(28):
                    shared_x[i][j][k]=data_x[i][j*28+k]
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

class LeNetConvPoolLayer(object):
   def __init__(self, rng, batch, n_new_feature, n_old_feature, filter_size, image_size, poolsize):
      filter_shape = (n_new_feature, n_old_feature, filter_size[0], filter_size[1])
      b_shape = (n_new_feature, image_size[0] - filter_size[0] + 1, image_size[1] - filter_size[1] + 1)
      fan_in = n_old_feature * filter_size[0] * filter_size[1]
      fan_out = n_new_feature * filter_size[0] * filter_size[1] // (poolsize[0] * poolsize[1])
      W_bound = numpy.sqrt(6. / (fan_in + fan_out))
      self.W = numpy.asarray(
         rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
         dtype='float64'
      )
      self.delta_W = numpy.zeros(filter_shape, dtype='float64')
      self.b = numpy.zeros(b_shape, dtype='float64')
      self.delta_b = numpy.zeros(b_shape, dtype='float64')
      cov_shape = (batch, n_new_feature, image_size[0] - filter_size[0] + 1, image_size[1] - filter_size[1] + 1)
      self.delta_conv = numpy.zeros(cov_shape, dtype='float64')
      self.z = numpy.zeros(cov_shape, dtype='float64')
      self.a_before_pooling = numpy.zeros(cov_shape, dtype='float64')
      self.poolsize = poolsize
      pool_shape = (batch, n_new_feature, cov_shape[2] // poolsize[0], cov_shape[3] // poolsize[1])
      self.delta_pooling = numpy.zeros(pool_shape, dtype='float64')
      self.id_max = numpy.zeros(pool_shape, dtype='float64')
      self.a_after_pooling = numpy.zeros(pool_shape, dtype='float64')
      self.batch = batch

   def forward_convolution(self, x):
      for i in range(x.shape[0]):
         for j in range(self.W.shape[0]):
            tmp = numpy.zeros((x.shape[2] - self.W.shape[2] + 1, x.shape[3] - self.W.shape[3] + 1))
            for k in range(self.W.shape[1]):
               tmp += signal.fftconvolve(x[i][k], self.W[j][k], 'valid')
            self.z[i][j] = tmp + self.b[j]
      self.a_before_pooling = (numpy.exp(self.z) - numpy.exp(-self.z)) / (numpy.exp(self.z) + numpy.exp(-self.z))

   def forward_pooling(self):
      tmp = numpy.zeros((self.a_before_pooling.shape[0], self.a_before_pooling.shape[1],
                         self.a_after_pooling.shape[2], self.a_after_pooling.shape[3],
                         self.poolsize[0] * self.poolsize[1]),
                        dtype='float64')
      for i in range(self.a_before_pooling.shape[0]):
         for j in range(self.a_before_pooling.shape[1]):
            for k in range(self.a_before_pooling.shape[2]):
               for l in range(self.a_before_pooling.shape[3]):
                  a=k // self.poolsize[0]
                  b=l // self.poolsize[1]
                  c=(k % self.poolsize[0]) * self.poolsize[1] + l %self.poolsize[1]
                  if (a>=tmp.shape[2])|(b>=tmp.shape[3]):
                      continue
                  tmp[i, j,a ,b ,c ] = self.a_before_pooling[i, j, k, l]
      self.a_after_pooling = tmp.max(axis=4)
      self.id_max = tmp.argmax(axis=4)

   def backward_comput_pooling_delta(self, next_delta, next_W):
      for i in range(self.a_after_pooling.shape[0]):
         for j in range(self.a_after_pooling.shape[1]):
            dummy = numpy.zeros((self.a_after_pooling.shape[2], self.a_after_pooling.shape[3]), dtype='float64')
            for k in range(next_delta.shape[1]):
               dummy += signal.fftconvolve(next_delta[i, k],numpy.rot90(next_W[k, j],2), 'full')
            self.delta_pooling[i][j] = dummy

   def backforward_update_W_b(self, x, learning_rate):
      for i in range(self.batch):
         for j in range(self.delta_conv.shape[1]):
            for k in range(self.delta_conv.shape[2]):
               for l in range(self.delta_conv.shape[3]):
                  a=k//self.poolsize[0]
                  b=l//self.poolsize[1]
                  c=(k % self.poolsize[0]) * self.poolsize[1] + l %self.poolsize[1]
                  if (a<self.delta_pooling.shape[2])&(b<self.delta_pooling.shape[3])&(c ==self.id_max[i][j][a][b]):
                     self.delta_conv[i][j][k][l] = self.delta_pooling[i][j][a][b]
                  else:
                     self.delta_conv[i][j][k][l] = 0
      self.delta_conv *= (1 - self.a_before_pooling ** 2)
      for j in range(self.W.shape[0]):
         for k in range(self.W.shape[1]):
            tmp=numpy.zeros((self.W.shape[2],self.W.shape[3]))
            for i in range(self.batch):
               tmp += signal.fftconvolve(x[i][k], self.delta_conv[i][j], 'valid')
            self.delta_W[j][k]=tmp/self.batch
      for j in range(self.delta_conv.shape[1]):
         temp = numpy.zeros((self.delta_b.shape[1], self.delta_b.shape[2]))
         for i in range(self.delta_conv.shape[0]):
            temp += self.delta_conv[i][j]
         self.delta_b[j] = temp/self.batch

      self.b -= learning_rate * self.delta_b
      self.W -= learning_rate * self.delta_W
      return self.delta_W,self.delta_conv


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
    def __init__(self,batch,rng,n_in,n_list_hidden_nodes,n_out):
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
        tt = numpy.dot(next_delta, next_W.transpose())
        return tt


class cnn(object):
    def __init__(self, batch, rng, image_a, image_b, feature=[5, 10], n_list_hidden_nodes=[50], pool_size=(2, 2),
                 filter_size=(5, 5), n_out=10):
        self.cov=[]
        n1=1
        n2=feature[0]
        now_image_a=image_a
        now_image_b=image_b
        for i in range(len(feature)):
            self.cov.append(LeNetConvPoolLayer(rng,batch,n2,n1,filter_size,(now_image_a,now_image_b),pool_size))
            now_image_a=(now_image_a-filter_size[0]+1)//pool_size[0]
            now_image_b=(now_image_b-filter_size[1]+1)//pool_size[1]
            if i!=len(feature)-1:
                n1=n2
                n2=feature[i+1]
        mpl_in=now_image_a*now_image_b*feature[-1]
        self.mlp=MLP(batch,rng,mpl_in,n_list_hidden_nodes,n_out)
        self.mpl_input=numpy.zeros((batch,mpl_in))
        self.batch=batch

    def feedforward(self,x):
        xx=numpy.zeros((self.batch,1,x.shape[1],x.shape[2]),dtype='float64')
        for i in range(self.batch):
            xx[i][0]=x[i]
        for i in range(len(self.cov)):
            self.cov[i].forward_convolution(xx)
            self.cov[i].forward_pooling()
            xx=self.cov[i].a_after_pooling
        for i in range(xx.shape[0]):
            vector=[]
            for j in range(xx.shape[1]):
                for k in range(xx.shape[2]):
                    for l in range(xx.shape[3]):
                        vector.append(xx[i][j][k][l])
            self.mpl_input[i]=numpy.asarray(vector)
        self.mlp.feedforward(self.mpl_input)

    def backpropagation(self,x,y,learning_rate,L2_reg):
        mpl_delta=self.mlp.backpropagation(self.mpl_input,y,learning_rate,L2_reg)
        for i in range(self.cov[-1].delta_pooling.shape[0]):
            for j in range(self.cov[-1].delta_pooling.shape[1]):
                for k in range(self.cov[-1].delta_pooling.shape[2]):
                    for l in range(self.cov[-1].delta_pooling.shape[3]):
                        self.cov[-1].delta_pooling[i][j][k][l]=mpl_delta[i][(j*self.cov[-1].delta_pooling.shape[2]+k)*self.cov[-1].delta_pooling.shape[3]+l]
        self.cov[-1].delta_pooling*=(1-self.cov[-1].a_after_pooling**2)
        i=len(self.cov)-1
        while i>=0:
            if i!=0:
                xx=self.cov[i-1].a_after_pooling
            else:
                xx = numpy.zeros((self.batch, 1, x.shape[1], x.shape[2]), dtype='float64')
                for j in range(self.batch):
                    xx[j][0] = x[j]
            next_W, next_delta = self.cov[i].backforward_update_W_b(xx, learning_rate)
            if i==0:
                break
            self.cov[i-1].backward_comput_pooling_delta(next_delta,next_W)
            i-=1

def cal_loss(y_p,y):
    f=0
    y_cal=numpy.argmax(y_p,axis=1)
    for i in range(y_cal.shape[0]):
        if y[i]!=y_cal[i]:
            f+=1
    return f/y_cal.shape[0]

def test_mlp(learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=1,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size
    n_test_batches = test_set_x.shape[0] // batch_size

    n_valid_batches//=30
    n_test_batches//=30
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    rng = numpy.random.RandomState(1234)
    classifier = cnn(batch_size,rng,28,28)

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    patience = 10
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
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
                    p_y_give_x = classifier.mlp.output_layer.p_y_given_x
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
                    for index in range(n_test_batches):
                        x = test_set_x[index * batch_size:(index + 1) * batch_size]
                        y = test_set_y[index * batch_size:(index + 1) * batch_size]
                        classifier.feedforward(x)
                        p_y_give_x = classifier.mlp.output_layer.p_y_given_x
                        test_score += cal_loss(p_y_give_x, y)
                    test_score /= n_valid_batches

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            # if patience <= iter:
            #     done_looping = True
            #     break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    test_mlp()