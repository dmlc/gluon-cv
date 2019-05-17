# pylint: skip-file
"""Create quantized model from JSON files..."""
import os
import warnings
import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon import SymbolBlock

__all__ = ['mobilenet1_0_int8', 'resnet50_v1_int8',
           'ssd_300_vgg16_atrous_voc_int8', 'ssd_512_mobilenet1_0_voc_int8',
           'ssd_512_resnet50_v1_voc_int8', 'ssd_512_vgg16_atrous_voc_int8']

def _not_impl(*args, **kwargs):
    raise NotImplementedError("Not yet implemented for quantized models")

def _create_quantized_models(name, sym_prefix):
    def func(pretrained=False, tag=None, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
        r"""Quantized model.

        Parameters
        ----------
        pretrained : bool or str
            Boolean value controls whether to load the default pretrained weights for model.
            String value represents the hashtag for a certain version of pretrained weights.
        tag : str, default is None
            Optional length-8 sha1sum of parameter file. If `None`, best parameter file
            will be used.
        ctx : Context, default CPU
            The context in which to load the pretrained weights.
        root : str, default $MXNET_HOME/models
            Location for keeping the model parameters.
        """
        from ..model_zoo import get_model
        from ..model_store import get_model_file
        curr_dir = os.path.abspath(os.path.dirname(__file__))
        model_name = name.replace('mobilenet1_', 'mobilenet1.')
        model_name = model_name.replace('mobilenet0_', 'mobilenet0.')
        json_file = os.path.join(curr_dir, '{}-symbol.json'.format(model_name))
        base_name = '_'.join(model_name.split('_')[:-1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            param_file = get_model_file(base_name, tag=tag, root=root) if pretrained else None
            net = get_model('_'.join(model_name.split('_')[:-1]), prefix=sym_prefix)
            classes = getattr(net, 'classes', [])
            sym_net = SymbolBlock.imports(json_file, ['data'], None, ctx=ctx)
            if param_file:
                # directly imports weights saved by save_parameters is not applicable
                # so we hack it by load and export once to a temporary params file
                import tempfile
                net.load_params(param_file)
                net.hybridize()
                if '512' in base_name:
                    net(mx.nd.zeros((1, 3, 512, 512)))
                elif '300' in base_name:
                    net(mx.nd.zeros((1, 3, 300, 300)))
                else:
                    net(mx.nd.zeros((1, 3, 224, 224)))
                with tempfile.TemporaryDirectory() as tmpdirname:
                    prefix = os.path.join(tmpdirname, 'tmp')
                    net.export(prefix, epoch=0)
                    param_prefix = prefix + '-0000.params'
                    sym_net.collect_params().load(param_prefix)
        sym_net.classes = classes
        sym_net.reset_class = _not_impl
        sym_net.set_nms = _not_impl
        return sym_net
    func.__name__ = name
    globals()[name] = func

_create_quantized_models('mobilenet1_0_int8', 'mobilenet0_')
_create_quantized_models('resnet50_v1_int8', 'resnetv10_')
_create_quantized_models('ssd_300_vgg16_atrous_voc_int8', 'ssd0_')
_create_quantized_models('ssd_512_mobilenet1_0_voc_int8', 'ssd0_')
_create_quantized_models('ssd_512_resnet50_v1_voc_int8', 'ssd0_')
_create_quantized_models('ssd_512_vgg16_atrous_voc_int8', 'ssd0_')
