"""Base Estimator"""
import os
import argparse
import copy
from datetime import datetime
from yacs.config import CfgNode

def set_default(default_config_func):
    def _apply(cls):
        # docstring
        cls.__doc__ = str(cls.__doc__) if cls.__doc__ else ''
        cls.__doc__ += ("\n\nParameters\n"
                        "----------\n"
                        "config : str, list or CfgNode\n"
                        "  Config used to override default configurations.\n"
                        "reporter : Reporter, default is `None`.\n"
                        "  If not `None`, reporter will be used to record metrics.\n"
                        "logdir : str, default is None.\n"
                        "  Directory for saving logs. If `None`, current working directory is used.\n")
        cls.__doc__ += '\nDefault configurations: \n----------\n'
        cls.__doc__ += str(default_config_func())
        # default config
        @classmethod
        def get_default_cfg(cls):
            return default_config_func()
        cls._get_default_config = get_default_cfg
        return cls
    return _apply

def config_to_args(parser, config, prefix=None):
    if prefix is None:
        prefix = []
    for k, v in sorted(config.items()):
        p = prefix + [str(k)]
        if isinstance(v, CfgNode):
            config_to_args(parser, v, prefix=p)
        else:
            parser.add_argument('--' + '.'.join(p), default=v, type=type(v), metavar="'" + str(v) + "'")
    return parser

def fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__

class BaseEstimator(object):
    # docstring of BaseEstimator is deliberately left empty
    def __init__(self, config=None, reporter=None, logdir=None):
        self.config = self._get_default_config()
        self._update_config(config)
        self.reporter = reporter
        self.logdir = os.path.abspath(logdir) if logdir else os.getcwd()

    def _update_config(self, config=None):
        if not config:
            return
        if isinstance(config, str) and os.path.splitext(config) in ('yml', 'yaml'):
            self.config.merge_from_file(config)
        elif isinstance(config, (list, tuple)):
            self.config.merge_from_list(list(config))
        elif isinstance(config, CfgNode):
            self.config.merge_from_other_cfg(config)
        else:
            raise ValueError('Unsupported config type: {0}.'.format(type(config)))

    def parse_args(self, args=None, namespace=None):
        parser = argparse.ArgumentParser('Estimator Arguments')
        parser = config_to_args(parser, self.config)
        args = parser.parse_args(args=args, namespace=namespace)
        list_config = []
        for k in vars(args):
            v = getattr(args, k)
            if isinstance(v, (tuple, list)):
                v = str(v)
            list_config += [k, v]
        self.config.merge_from_list(list_config)

    def finalize_config(self):
        self.config.freeze()
        fn = fullname(self) + datetime.now().strftime("-%m-%d-%Y-%H-%M-%S.yml")
        fn = os.path.join(self.logdir, fn)
        with open(fn, 'wt') as f:
            f.write('# ' + fn + '\n')
            f.write(str(self.config))
            f.write('\n')

    def fit(self, train_data):
        raise NotImplementedError

    def evaluate(self, val_data):
        raise NotImplementedError

    def predict(self, test_data):
        raise NotImplementedError

    def deploy(self):
        raise NotImplementedError
