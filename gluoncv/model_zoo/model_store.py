"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile

from mxnet.gluon.utils import check_sha1
from ..utils import download

_model_sha1 = {name: checksum for checksum, name in [
    ('4fa2e1ad96b8c8d1ba9e5a43556cd909d70b3985', 'vgg16_atrous'),
    ('0e169fbb64efdee6985c3c175ec4298c4bda0298', 'ssd_300_vgg16_atrous_voc'),
    ('daf8181b615b480236fcb8474545077891276945', 'ssd_512_vgg16_atrous_voc'),
    ('9c8b225a552614e4284a0f647331bfdc6940eb4a', 'ssd_512_resnet50_v1_voc'),
    ('2cc0f93edf1467f428018cc7261d3246dfa15259', 'ssd_512_resnet101_v2_voc'),
    ('37c180765a4eb3e67751d6bacac47bb9156f5fff', 'ssd_512_mobilenet1.0_voc'),
    ('9b9ee04e4fba85431d11b9f7eb3a232b6606c7f6', 'faster_rcnn_resnet50_v2a_voc'),
    ('b302ad8a8660345c368448141d8acf30b5a3801d', 'ssd_300_vgg16_atrous_coco'),
    ('5c86064290c05eccbdd88475376c71c595c8325c', 'ssd_512_vgg16_atrous_coco'),
    ('c48351620d4f0cbc49e4f7a84c8e67ef8fdc6e09', 'ssd_512_resnet50_v1_coco'),
    ('78e5daa9e47b1f2f9c3ffd3ffb9714c8d968a367', 'faster_rcnn_resnet50_v1b_coco'),
    ('121e1579d811b091940b3b1fa033e1f0d1dca40f', 'cifar_resnet20_v1'),
    ('4f2d18804c94f2d283b8b45256d048bd3d6dd479', 'cifar_resnet20_v2'),
    ('2fb251e60babdceb103e9659b3baa0dea20a14d7', 'cifar_resnet56_v1'),
    ('0a3e74104ec7bcfffefe2d9d5cc1f8e74311ec51', 'cifar_resnet56_v2'),
    ('a0e1f860475bf5369f6da07e0c2e03a4ae9cff9c', 'cifar_resnet110_v1'),
    ('bf160f8b3cb3884a1ea871739f3c8e151e114159', 'cifar_resnet110_v2'),
    ('7c07b5ba6e850f9c37ca1e57c0a2e529455cc2e4', 'cifar_wideresnet16_10'),
    ('4a3466aadd4c3ddbcb968bca862d0e59d6f15ec1', 'cifar_wideresnet28_10'),
    ('085ca2afabbe0ddfe87d0edc5408bcfcfbffd414', 'cifar_wideresnet40_8'),
    ('e8ff9f4f9cb319dfbf524d01e487af9a7f8a3cf5', 'cifar_resnext29_16x64d'),
    ('2d9d980c990442f826f20781ed039851e78dabe3', 'resnet18_v1b'),
    ('8e16b84814e84f64d897854003f049872991eaa6', 'resnet34_v1b'),
    ('e263a9860be0a373003d011564f10701d4954fb8', 'resnet50_v1b'),
    ('c9d451fc69773007958205659bad43c9ae77e20d', 'resnet101_v1b'),
    ('e74027961d155170868f30bd3f113d7feb44f618', 'resnet152_v1b'),
    ('953657f235cc52dbc60f3874f9d437c380045cd0', 'fcn_resnet50_voc'),
    ('70a6f22a1a0b6ddd1f680de587d67b5c2c0acc0b', 'fcn_resnet101_voc'),
    ('b1b11976bf753ed1e05a526065e1666950fcf0a2', 'fcn_resnet50_ade'),
    ('3133bd42540ffee84a54e95468de74560a4a27b9', 'psp_resnet50_ade'),
    ('b8f9c5f193d84b828fa78d7bd7646ca7f11a29bc', 'resnet50_v2a'),
    ]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, root=os.path.join('~', '.mxnet', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    from mxnet.gluon.model_zoo.model_store import get_model_file as upstream_get_model_file
    try:
        file_name = upstream_get_model_file(name=name, root=root)
    except ValueError:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    sha1_hash = _model_sha1[name]
    if os.path.exists(file_path):
        if check_sha1(file_path, sha1_hash):
            return file_path
        else:
            print('Mismatch in the content of model file detected. Downloading again.')
    else:
        print('Model file is not found. Downloading.')

    if not os.path.exists(root):
        os.makedirs(root)

    zip_file_path = os.path.join(root, file_name+'.zip')
    repo_url = os.environ.get('MXNET_GLUON_REPO', apache_repo_url)
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name=file_name),
             path=zip_file_path,
             overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    if check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')

def purge(root=os.path.join('~', '.mxnet', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".params"):
            os.remove(os.path.join(root, f))

def pretrained_model_list():
    return list(_model_sha1.keys())
