'''Functions for modified initialization'''

import mxnet as mx

def init_mod_sqnet(net, ctx):
    '''
    Modified initialization for SqueezeNet.
    Initialize final conv layer to random normal weights by doing one forward pass through the net and set weights explicitly.
    Parameters
    ----------
    net: SqueezeNet HybridBlock from Gluon model zoo initialized with mx.init.MSRAPrelu()
    Returns
    -------
    net: Input net with modified initialization
    '''
    data = [mx.nd.zeros(shape=(1,3,224,224), ctx=i) for i in ctx]
    outputs = [net(X) for X in data]
    new_init_w = mx.nd.random_normal(shape=(1000, 512, 1, 1), scale=0.01)
    final_conv_layer_params = net.output[0].params
    final_conv_layer_params.get('weight').set_data(new_init_w)
    return net        
