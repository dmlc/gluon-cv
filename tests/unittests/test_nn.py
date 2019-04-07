import mxnet as mx
from gluoncv.nn import GroupNorm

def test_groupnorm():
    ctx=mx.context.current_context()
    x = mx.nd.random.uniform(1, 2, (4, 16, 8, 8), ctx=ctx)
    gn = GroupNorm(4, 16)
    gn.initialize(ctx=ctx)
    y = gn(x)
    y = y.reshape(0, 4, -1)
    print('y.mean(2) =', y.mean(2))
    mx.test_utils.assert_almost_equal(y.mean(2).asnumpy(),
                                      mx.nd.zeros_like(y.mean(2)).asnumpy(),
                                      rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    import nose
    nose.runmodule()
