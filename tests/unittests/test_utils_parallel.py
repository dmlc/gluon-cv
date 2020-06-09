import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
from gluoncv.utils.parallel import DataParallelModel, DataParallelCriterion

def test_data_parallel():
    # test gluon.contrib.parallel.DataParallelModel
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(nn.Conv2D(in_channels=1, channels=5, kernel_size=5))
        net.add(nn.Activation('relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        net.add(nn.Conv2D(in_channels=5, channels=5, kernel_size=5))
        net.add(nn.Activation('relu'))
        net.add(nn.MaxPool2D(pool_size=2, strides=2))
        # The Flatten layer collapses all axis, except the first one, into one axis.
        net.add(nn.Flatten())
        net.add(nn.Dense(8,in_units=80))
        net.add(nn.Activation('relu'))
        net.add(nn.Dense(10, in_units=8))

    net.collect_params().initialize()
    criterion = gluon.loss.SoftmaxCELoss(axis=1)

    def test_net_sync(net, criterion, sync, nDevices):
        ctx_list = [mx.cpu(0) for i in range(nDevices)]
        net = DataParallelModel(net, ctx_list, sync=sync)
        criterion = DataParallelCriterion(criterion, ctx_list, sync=sync)
        iters = 10
        bs = 2
        # train mode
        for i in range(iters):
            x = mx.random.uniform(shape=(bs, 1, 28, 28))
            t = nd.ones(shape=(bs))
            with autograd.record():
                y = net(x)
                loss = criterion(y, t)
                autograd.backward(loss)
        # evaluation mode
        for i in range(iters):
            x = mx.random.uniform(shape=(bs, 1, 28, 28))
            y = net(x)
        nd.waitall()

    # test_net_sync(net, criterion, True, 1)
    test_net_sync(net, criterion, True, 2)
    # test_net_sync(net, criterion, False, 1)
    test_net_sync(net, criterion, False, 2)


if __name__ == "__main__":
    import nose
    nose.runmodule()
