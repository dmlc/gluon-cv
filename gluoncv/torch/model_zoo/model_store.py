"""Model store which provides pretrained models."""
from __future__ import print_function

import os
import logging
import portalocker

from ..utils.model_utils import download, check_sha1


__all__ = ['get_model_file', 'purge']


_model_sha1 = {name: checksum for checksum, name in [
    ('854b23e460ab1fbc5a5cd458b439a90be1f98119', 'resnet18_v1b_kinetics400'),
    ('124a2fa45fce7102243515bdebfb262728c10ad0', 'resnet34_v1b_kinetics400'),
    ('9939dbdfdbf11cb42c06456f341596e75becf074', 'resnet50_v1b_kinetics400'),
    ('172afa3bcc5e922962c335b6ea9b7cedc8132c61', 'resnet101_v1b_kinetics400'),
    ('3dedb835f333e234a9279bbef4a9e2e840acc032', 'resnet152_v1b_kinetics400'),
    ('cbb9167bd293a82475f5f179a29de1f5c2665b8a', 'resnet50_v1b_sthsthv2'),
    ('185454973645ff66b6310e715863384db135d87f', 'i3d_resnet50_v1_kinetics400'),
    ('a9bb4f89551c30d553a487fcc560ff7f41a3de8c', 'i3d_resnet101_v1_kinetics400'),
    ('9df1e103d54bc1227da58d19347e1371cf98a9fb', 'i3d_nl5_resnet50_v1_kinetics400'),
    ('281e1e8a8ef78ada7e095b43caf0cd3f75e7a6a4', 'i3d_nl10_resnet50_v1_kinetics400'),
    ('2cea8edd07e5565b9c1eb66ca862ade3127ce523', 'i3d_nl5_resnet101_v1_kinetics400'),
    ('526a2ed07cffeab6d827ab28c979b537100cd2b3', 'i3d_nl10_resnet101_v1_kinetics400'),
    ('e975d989c1eb1e66e5be36c1c4d6ff626708cfae', 'i3d_resnet50_v1_sthsthv2'),
    ('1d1eadb258ccfc0c117db0ff0eaf1638d18405b3', 'slowfast_4x16_resnet50_kinetics400'),
    ('e94e9a57eaa2f4dde12673bc6632f9aa090d8c21', 'slowfast_8x8_resnet50_kinetics400'),
    ('db5e9fef55b28eb55b6dc7b2cba98d0c14fc3568', 'slowfast_8x8_resnet101_kinetics400'),
    ('0520323165727d59002c6c62071bc1dc28248469', 'slowfast_16x8_resnet50_sthsthv2'),
    ('078c817bf18d702891de553de61c9115e27c4afd', 'i3d_slow_resnet50_f32s2_kinetics400'),
    ('a3e419f1b9cde3b57b9b19a9bd93281bb819a871', 'i3d_slow_resnet50_f16s4_kinetics400'),
    ('b5be1a2e32158748a7efc675081affd32006fe9a', 'i3d_slow_resnet101_f16s4_kinetics700'),
    ('1c3d98a1831ee477def36d6085635c2860db0b77', 'i3d_slow_resnet50_f8s8_kinetics400'),
    ('db37cd518eab5f1cc394d0f44117c392d0174602', 'i3d_slow_resnet101_f32s2_kinetics400'),
    ('cb6b78d9e58f7724efb63f705be7c8a12fe8f5ce', 'i3d_slow_resnet101_f16s4_kinetics400'),
    ('82e399c1114016126645c5e3e67bad164ba3ae84', 'i3d_slow_resnet101_f8s8_kinetics400'),
    ('340a59522b4ffe7e3621ad4a269e4cd4a99f74b2', 'r2plus1d_v1_resnet18_kinetics400'),
    ('5102fd1736a2a205f5fd7bded5d2e0d2e5ca6307', 'r2plus1d_v1_resnet34_kinetics400'),
    ('9a3b665c182f81f22d4a105c75fac030ad20a628', 'r2plus1d_v1_resnet50_kinetics400'),
    ('42707ffcab518cdda93523004360de167863c5d8', 'r2plus1d_v2_resnet152_kinetics400'),
    ('82855d2c85a888e96477130253b90a4892bdc649', 'ircsn_v2_resnet152_f32s2_kinetics400'),
    ('368108eb6bca9143318937319c3efec09e0419af', 'tpn_resnet50_f8s8_kinetics400'),
    ('6bf899df92224d3c0c117c940ef0c95d51fedb26', 'tpn_resnet50_f16s4_kinetics400'),
    ('27710ce8091a317f50fc55e171b29c096b3d253e', 'tpn_resnet50_f32s2_kinetics400'),
    ('092c2f7fc98ec9ea204487fa074bc0ca832aba9a', 'tpn_resnet101_f8s8_kinetics400'),
    ('647080df950480e6544e6cd471f0024125656593', 'tpn_resnet101_f16s4_kinetics400'),
    ('a94422a94acda20f34eb07604cd3315307d1a9a2', 'tpn_resnet101_f32s2_kinetics400'),
    ('0bac43a266b8927685190597610998d3e4ecc53c', 'directpose_resnet50_lpf_fpn_coco'),
]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}torch/models/{file_name}.pth'


def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]


def get_model_file(name, tag=None, root=os.path.join('~', '.torch', 'models')):
    r"""Return location for the pretrained on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    name : str
        Name of the model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    if 'TORCH_HOME' in os.environ:
        root = os.path.join(os.environ['TORCH_HOME'], 'models')

    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=short_hash(name))
    root = os.path.expanduser(root)
    params_path = os.path.join(root, file_name + '.pth')
    lockfile = os.path.join(root, file_name + '.lock')
    if use_tag:
        sha1_hash = tag
    else:
        sha1_hash = _model_sha1[name]

    if not os.path.exists(root):
        os.makedirs(root)

    with portalocker.Lock(lockfile, timeout=int(os.environ.get('GLUON_MODEL_LOCK_TIMEOUT', 300))):
        if os.path.exists(params_path):
            if check_sha1(params_path, sha1_hash):
                return params_path
            else:
                logging.warning("Hash mismatch in the content of model file '%s' detected. "
                                "Downloading again.", params_path)
        else:
            logging.info('Model file not found. Downloading.')

        zip_file_path = os.path.join(root, file_name + '.pth')
        repo_url = os.environ.get('TORCH_GLUON_REPO', apache_repo_url)
        if repo_url[-1] != '/':
            repo_url = repo_url + '/'
        download(_url_format.format(repo_url=repo_url, file_name=file_name),
                 path=zip_file_path,
                 overwrite=True)
        # Make sure we write the model file on networked filesystems
        try:
            os.sync()
        except AttributeError:
            pass
        if check_sha1(params_path, sha1_hash):
            return params_path
        else:
            raise ValueError('Downloaded file has different hash. Please try again.')


def purge(root=os.path.join('~', '.torch', 'models')):
    r"""Purge all pretrained model files in local file store.

    Parameters
    ----------
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    root = os.path.expanduser(root)
    files = os.listdir(root)
    for f in files:
        if f.endswith(".pth"):
            os.remove(os.path.join(root, f))
