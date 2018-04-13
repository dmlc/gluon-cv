"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile

from mxnet.gluon.utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('ce85f11ce208d8f5adc9da52753b717db73a2c6e', 'cifar_resnet20_v1'),
    ('f920e8b8186132e64518a4502c57e381ef336b72', 'cifar_resnet20_v2'),
    ('aa01e54160e2c92b8dee2319f1c2a8ea0e327ed1', 'cifar_resnet56_v1'),
    ('81edd76526cc736df96b165eb6f8850bcad9d6d6', 'cifar_resnet56_v2'),
    ('3a9d8dda8e54f06735b1d61b85002d00c8cd5005', 'cifar_resnet110_v1'),
    ('d7bacf7cb1b5e19348941e6d486b9262efe78b15', 'cifar_resnet110_v2'),
    ('ebc52a3cb68fb9a14f59944b510cc682675a70cc', 'cifar_wideresnet16_10'),
    ('0ef2c7bec9c2c48b8ff52440ae84869ce2db20b1', 'cifar_wideresnet28_10'),
    ('5369ff00708900aa34c85088ac2c15fd42f0b243', 'cifar_wideresnet40_8')]}

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
