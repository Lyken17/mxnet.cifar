import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd

alex_net = gluon.nn.HybridSequential()
with alex_net.name_scope():
    #  First convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4, 4), activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    #  Second convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2, 2)))
    # Third convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fourth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fifth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))
    # Flatten and apply fullly connected layers
    alex_net.add(gluon.nn.Flatten())
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(10))

net = alex_net
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()

################## GPUs #################
GPU_COUNT = 3  # increase if you have more
batch_size = 128 * GPU_COUNT
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
# ctx = [mx.cpu()]

net.collect_params().initialize(ctx=ctx)


def forward_backward(net, data, label):
    with autograd.record():
        losses = [loss(net(X), Y) for X, Y in zip(data, label)]
    for l in losses:
        l.backward()


################## TEST ##################
# from mxnet.test_utils import get_mnist
# mnist = get_mnist()
# batch = mnist['train_data'][0:GPU_COUNT*2, :]
# data = gluon.utils.split_and_load(batch, ctx)
# print(net(data[0]))
# print(net(data[1]))
#
#
# weight = net.collect_params()['cnn_conv0_weight']
#
# for c in ctx:
#     print('=== channel 0 of the first conv on {} ==={}'.format(
#         c, weight.data(ctx=c)[0]))
#
#
# label = gluon.utils.split_and_load(mnist['train_label'][0:4], ctx)
# forward_backward(net, data, label)
# for c in ctx:
#     print('=== grad of channel 0 of the first conv2d on {} ==={}'.format(
#         c, weight.grad(ctx=c)[0]))


from mxnet.io import NDArrayIter
from time import time


def train_batch(batch, ctx, net, trainer):
    # split the data batch and load them on GPUs
    data = gluon.utils.split_and_load(batch[0], ctx)
    label = gluon.utils.split_and_load(batch[1], ctx)
    # compute gradient
    forward_backward(net, data, label)
    # update parameters
    trainer.step(batch[0].shape[0])


def valid_batch(batch, ctx, net):
    data = batch[0].as_in_context(ctx)
    pred = nd.argmax(net(data), axis=1)
    return nd.sum(pred.astype('int32') == batch[1].as_in_context(ctx).astype('int32')).asscalar()


def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2, 0, 1))
    data = data.astype(np.float32)
    return data, label


def run(num_gpus, batch_size, lr):
    # the list of GPUs will be used
    # ctx = [mx.gpu(i) for i in range(num_gpus)]
    # print('Running on {}'.format(ctx))
    # ctx = mx.cpu()
    # data iterator
    # mnist = get_mnist()
    # train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size)
    # valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size)

    ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
    # batch_size = 64
    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True, transform=transformer),
        batch_size=batch_size, shuffle=True, last_batch='discard')

    valid_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False, transform=transformer),
        batch_size=batch_size, shuffle=False, last_batch='discard')

    print('Batch size is {}'.format(batch_size))

    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(5):
        # train
        start = time()
        # train_data.reset()
        total = len(train_data)
        for idx, batch in enumerate(train_data):
            train_batch(batch, ctx, net, trainer)
            print("Iter [%d/%d]" % (idx, total))
        nd.waitall()  # wait until all computations are finished to benchmark the time
        print('Epoch %d, training time = %.1f sec' % (epoch, time() - start))

        # validating
        # valid_data.reset()
        total = len(valid_data)
        correct, num = 0.0, 0.0
        for idx, batch in enumerate(valid_data):
            correct += valid_batch(batch, ctx, net)
            num += batch[0].shape[0]
            print('Valid[%d/%d] accuracy = %.4f' % (idx, total, correct / num))
        print('Epoch %d validation accuracy = %.4f' % (epoch, correct / num))


# run(1, batch_size, .3)
run(GPU_COUNT, batch_size, .1 * GPU_COUNT)
