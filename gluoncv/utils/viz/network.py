"""Visualize network structure"""
import tempfile
try:
    import graphviz
except ImportError:
    graphviz = None
import mxnet as mx
from mxnet import gluon

def plot_network(block, shape=(1, 3, 224, 224), save_prefix=None):
    """Plot network to visualize internal structures.

    Parameters
    ----------
    block : mxnet.gluon.HybridBlock
        A hybridizable network to be visualized.
    shape : tuple of int
        Desired input shape, default is (1, 3, 224, 224).
    save_prefix : str or None
        If not `None`, will save rendered pdf to disk with prefix.

    """
    if graphviz is None:
        raise RuntimeError("Cannot import graphviz.")
    if not isinstance(block, gluon.HybridBlock):
        raise ValueError("block must be HybridBlock, given {}".format(type(block)))
    data = mx.sym.var('data')
    sym = block(data)
    if isinstance(sym, tuple):
        sym = mx.sym.Group(sym)
    a = mx.viz.plot_network(sym, shape={'data':shape},
                            node_attrs={'shape':'rect', 'fixedsize':'false'})
    a.view(tempfile.mktemp('.gv'))
    if isinstance(save_prefix, str):
        a.render(save_prefix)

def plot_mxboard(block, logdir='./logs'):
    """Plot network to visualize internal structures.

    Parameters
    ----------
    block : mxnet.gluon.HybridBlock
        A hybridizable network to be visualized.
    logdir : str
        The directory to save.

    """
    try:
        from mxboard import SummaryWriter
    except ImportError:
        print('mxboard is required. Please install via `pip install mxboard` ' +
              'or refer to https://github.com/awslabs/mxboard.')
        raise
    data = mx.sym.var('data')
    sym = block(data)
    if isinstance(sym, tuple):
        sym = mx.sym.Group(sym)
    with SummaryWriter(logdir=logdir) as sw:
        sw.add_graph(sym)
    usage = '`tensorboard --logdir={} --host=127.0.0.1 --port=8888`'.format(logdir)
    print('Log saved. Use: {} to visualize it'.format(usage))
