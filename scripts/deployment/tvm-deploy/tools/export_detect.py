"""Helper utils for export HybridBlock to symbols."""
import mxnet as mx
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn

import argparse
parser = argparse.ArgumentParser(description='Export gcv object detection models using TVM')
parser.add_argument('--model', type=str, help='model name')
args = parser.parse_args()


class MyPreprocess(HybridBlock):
    def __init__(self, **kwargs):
        super(MyPreprocess, self).__init__(**kwargs)
        with self.name_scope():
            mean = mx.nd.array([123.675, 116.28, 103.53]).reshape((1, 3, 1, 1))
            scale = mx.nd.array([58.395, 57.12, 57.375]).reshape((1, 3, 1, 1))
            self.init_mean = self.params.get_constant('init_mean', mean)
            self.init_scale = self.params.get_constant('init_scale', scale)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, init_mean, init_scale):
        x = F.broadcast_minus(x, init_mean)
        x = F.broadcast_div(x, init_scale)
        return x

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
                if tuner == 'xgb' or tuner == 'xgb-rank':
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
                n_trial=len(task.config_space)
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
            graph, lib, params = relay.build(
                sym, target, params=params)

    # export library, json graph and parameters
    lib.export_library(path + '_lib.so')
    with open(path + '_graph.json', 'w') as fo:
        fo.write(graph)
    with open(path + '_{:04n}.params'.format(epoch), 'wb') as fo:
        fo.write(relay.save_param_dict(params))
        
if __name__ == '__main__':
    import gluoncv as gcv
    model = gcv.model_zoo.get_model(args.model, pretrained=True)
    export_tvm("model", model, (3,512,605), ctx=mx.cpu(), preprocess=MyPreprocess(), target='llvm -mcpu=core-avx2', use_autotvm=False)
    