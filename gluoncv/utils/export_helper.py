"""Helper utils for export HybridBlock to symbols."""
from __future__ import absolute_import
import copy
import mxnet as mx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


class _DefaultPreprocess(HybridBlock):
    """Default preprocess block used by GluonCV.

    The default preprocess block includes:

        - mean [123.675, 116.28, 103.53]

        - std [58.395, 57.12, 57.375]

        - transpose to (B, 3, H, W)

    It is used to transform from resized original images with shape (1, H, W, 3) or (B, H, W, 3)
    in range (0, 255) and RGB color format.

    """
    def __init__(self, **kwargs):
        super(_DefaultPreprocess, self).__init__(**kwargs)
        with self.name_scope():
            mean = mx.nd.array([123.675, 116.28, 103.53]).reshape((1, 1, 1, 3))
            scale = mx.nd.array([58.395, 57.12, 57.375]).reshape((1, 1, 1, 3))
            self.init_mean = self.params.get_constant('init_mean', mean)
            self.init_scale = self.params.get_constant('init_scale', scale)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, init_mean, init_scale):
        x = F.broadcast_minus(x, init_mean)
        x = F.broadcast_div(x, init_scale)
        x = F.transpose(x, axes=(0, 3, 1, 2))
        return x

def export_block(path, block, data_shape=None, epoch=0, preprocess=True, layout='HWC',
                 ctx=mx.cpu()):
    """Helper function to export a HybridBlock to symbol JSON to be used by
    `SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface..

    Parameters
    ----------
    path : str
        Path to save model.
        Two files path-symbol.json and path-xxxx.params will be created,
        where xxxx is the 4 digits epoch number.
    block : mxnet.gluon.HybridBlock
        The hybridizable block. Note that normal gluon.Block is not supported.
    data_shape : tuple of int, default is None
        Fake data shape just for export purpose, in format (H, W, C) for 2D data
        or (T, H, W, C) for 3D data.
        If you don't specify ``data_shape``, `export_block` will try use some common data_shapes,
        e.g., (224, 224, 3), (256, 256, 3), (299, 299, 3), (512, 512, 3)...
        If any of this ``data_shape`` goes through, the export will succeed.
    epoch : int
        Epoch number of saved model.
    preprocess : mxnet.gluon.HybridBlock, default is True.
        Preprocess block prior to the network.
        By default (True), it will subtract mean [123.675, 116.28, 103.53], divide
        std [58.395, 57.12, 57.375], and convert original image (B, H, W, C and range [0, 255]) to
        tensor (B, C, H, W) as network input. This is the default preprocess behavior of all GluonCV
        pre-trained models.
        You can use custom pre-process hybrid block or disable by set ``preprocess=None``.
    layout : str, default is 'HWC'
        The layout for raw input data. By default is HWC. Supports 'HWC', 'CHW', 'THWC' and 'CTHW'.
        Note that image channel order is always RGB.
    ctx: mx.Context, default mx.cpu()
        Network context.

    Returns
    -------
    None

    """
    # input image layout
    layout = layout.upper()
    if data_shape is None:
        if layout == 'HWC':
            data_shapes = [(s, s, 3) for s in (224, 256, 299, 300, 320, 416, 512, 600)]
        elif layout == 'CHW':
            data_shapes = [(3, s, s) for s in (224, 256, 299, 300, 320, 416, 512, 600)]
        else:
            raise ValueError('Unable to predict data_shape, please specify.')
    else:
        data_shapes = [data_shape]

    # use deepcopy of network to avoid in-place modification
    copy_block = block
    if '_target_generator' in copy_block._children:
        copy_block = copy.deepcopy(block)
        copy_block._children.pop('_target_generator')

    # avoid using some optimized operators that are not yet available outside mxnet
    if 'box_encode' in mx.sym.contrib.__dict__:
        box_encode = mx.sym.contrib.box_encode
        mx.sym.contrib.__dict__.pop('box_encode')
    else:
        box_encode = None

    if preprocess:
        # add preprocess block
        if preprocess is True:
            assert layout == 'HWC', \
                "Default preprocess only supports input as HWC, provided {}.".format(layout)
            preprocess = _DefaultPreprocess()
        else:
            if not isinstance(preprocess, HybridBlock):
                raise TypeError("preprocess must be HybridBlock, given {}".format(type(preprocess)))
        wrapper_block = nn.HybridSequential()
        preprocess.initialize(ctx=ctx)
        wrapper_block.add(preprocess)
        wrapper_block.add(copy_block)
    else:
        wrapper_block = copy_block
        assert layout in ('CHW', 'CTHW'), \
            "Default layout is CHW for 2D models and CTHW for 3D models if preprocess is None," \
            + " provided {}.".format(layout)
    wrapper_block.collect_params().reset_ctx(ctx)

    # try different data_shape if possible, until one fits the network
    last_exception = None
    for dshape in data_shapes:

        if layout == 'HWC':
            h, w, c = dshape
            x = mx.nd.zeros((1, h, w, c), ctx=ctx)
        elif layout == 'CHW':
            c, h, w = dshape
            x = mx.nd.zeros((1, c, h, w), ctx=ctx)
        elif layout == 'THWC':
            t, h, w, c = dshape
            x = mx.nd.zeros((1, t, h, w, c), ctx=ctx)
        elif layout == 'CTHW':
            c, t, h, w = dshape
            x = mx.nd.zeros((1, c, t, h, w), ctx=ctx)
        else:
            raise RuntimeError('Input layout %s is not supported yet.' % (layout))

        # hybridize and forward once
        wrapper_block.hybridize()
        try:
            wrapper_block(x)
            wrapper_block.export(path, epoch)
            last_exception = None
            break
        except MXNetError as e:
            last_exception = e
    if last_exception is not None:
        raise RuntimeError(str(last_exception).splitlines()[0])

def export_tvm(path, block, data_shape, epoch=0, preprocess=True, layout='HWC',
               ctx=mx.cpu(), target='llvm', opt_level=3, use_autotvm=False):
    """Helper function to export a HybridBlock to TVM executable. Note that tvm package needs
    to be installed(https://tvm.ai/).

    Parameters
    ----------
    path : str
        Path to save model.
        Three files path_deploy_lib.tar, path_deploy_graph.json and path_deploy_xxxx.params
        will be created, where xxxx is the 4 digits epoch number.
    block : mxnet.gluon.HybridBlock
        The hybridizable block. Note that normal gluon.Block is not supported.
    data_shape : tuple of int, required
        Unlike `export_block`, `data_shape` is required here for the purpose of optimization.
        If dynamic shape is required, you can use the shape that most fits the inference tasks,
        but the optimization won't accommodate all situations.
    epoch : int
        Epoch number of saved model.
    preprocess : mxnet.gluon.HybridBlock, default is True.
        Preprocess block prior to the network.
        By default (True), it will subtract mean [123.675, 116.28, 103.53], divide
        std [58.395, 57.12, 57.375], and convert original image (B, H, W, C and range [0, 255]) to
        tensor (B, C, H, W) as network input. This is the default preprocess behavior of all GluonCV
        pre-trained models.
        You can use custom pre-process hybrid block or disable by set ``preprocess=None``.
    layout : str, default is 'HWC'
        The layout for raw input data. By default is HWC. Supports 'HWC' and 'CHW'.
        Note that image channel order is always RGB.
    ctx: mx.Context, default mx.cpu()
        Network context.
    target : str, default is 'llvm'
        Runtime type for code generation, can be ('llvm', 'cuda', 'opencl', 'metal'...)
    opt_level : int, default is 3
        TVM optimization level, if supported, higher `opt_level` may generate more efficient
        runtime library, however, some operator may not support high level optimization, which will
        fallback to lower `opt_level`.
    use_autotvm : bool, default is False
        Use autotvm for performance tuning. Note that this can take very long time, since it's a
        search and model based tuning process.

    Returns
    -------
    None

    """
    try:
        import tvm
        from tvm import autotvm
        from tvm import relay
        from tvm.relay import testing
        from tvm.autotvm.tuner import XGBTuner, RandomTuner
        import tvm.contrib.graph_runtime as runtime
    except ImportError:
        print("TVM package required, please refer https://tvm.ai/ for installation guide.")
        raise

    # add preprocess block if necessary
    if preprocess:
        # add preprocess block
        if preprocess is True:
            preprocess = _DefaultPreprocess()
        else:
            if not isinstance(preprocess, HybridBlock):
                raise TypeError("preprocess must be HybridBlock, given {}".format(type(preprocess)))
        wrapper_block = nn.HybridSequential()
        preprocess.initialize(ctx=ctx)
        wrapper_block.add(preprocess)
        wrapper_block.add(block)
    else:
        wrapper_block = block
    wrapper_block.collect_params().reset_ctx(ctx)

    # convert to relay graph
    sym, params = relay.frontend.from_mxnet(wrapper_block, shape={"data": data_shape})

    if use_autotvm:
        def tune_kernels(tasks,
                         measure_option,
                         tuner='gridsearch',
                         early_stopping=None,
                         log_filename='tuning.log'):
            for i, tsk in enumerate(tasks):
                prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

                # converting conv2d tasks to conv2d_NCHWc tasks
                op_name = tsk.workload[0]
                if op_name == 'conv2d':
                    func_create = 'topi_x86_conv2d_NCHWc'
                elif op_name == 'depthwise_conv2d_nchw':
                    func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
                else:
                    raise ValueError("Tuning {} is not supported on x86".format(op_name))

                task = autotvm.task.create(func_create, args=tsk.args,
                                           target=target, template_key='direct')
                task.workload = tsk.workload

                # create tuner
                if tuner in ('xgb', 'xgb-rank'):
                    tuner_obj = XGBTuner(task, loss_type='rank')
                elif tuner == 'ga':
                    tuner_obj = GATuner(task, pop_size=50)
                elif tuner == 'random':
                    tuner_obj = RandomTuner(task)
                elif tuner == 'gridsearch':
                    tuner_obj = GridSearchTuner(task)
                else:
                    raise ValueError("Invalid tuner: " + tuner)

                # do tuning
                n_trial = len(task.config_space)
                tuner_obj.tune(n_trial=n_trial,
                               early_stopping=early_stopping,
                               measure_option=measure_option,
                               callbacks=[
                                   autotvm.callback.progress_bar(n_trial, prefix=prefix),
                                   autotvm.callback.log_to_file(log_filename)])

        #
        tasks = autotvm.task.extract_from_program(sym, target=target,
                                                  params=params, ops=(relay.op.nn.conv2d,))
        logging.warning('Start tunning, this can be slow...')
        tuning_option = {
            'log_filename': 'tune.log',
            'tuner': 'random',
            'early_stopping': None,

            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(number=10, repeat=1,
                                           min_repeat_ms=1000),
            ),
        }
        tune_kernels(tasks, **tuning_option)

        with autotvm.apply_history_best(log_file):
            with relay.build_config(opt_level=opt_level):
                graph, lib, params = relay.build_module.build(
                    sym, target=target, params=params)

    else:
        with relay.build_config(opt_level=opt_level):
            graph, lib, params = relay.build_module.build(
                sym, target, params=params)

    # export library, json graph and parameters
    lib.export_library(path + '_deploy_lib.so')
    with open(path + '_deploy_graph.json', 'w') as fo:
        fo.write(graph)
    with open(path + '_deploy_{:04n}.params'.format(epoch), 'wb') as fo:
        try:
            fo.write(relay.compiler.save_param_dict(params))
        except AttributeError:
            fo.write(relay.save_param_dict(params))
