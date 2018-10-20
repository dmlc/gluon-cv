"""Model store which provides pretrained models."""
from __future__ import print_function
__all__ = ['get_model_file', 'purge']
import os
import zipfile

from ..utils import download, check_sha1

_model_sha1 = {name: checksum for checksum, name in [
    ('44335d1f0046b328243b32a26a4fbd62d9057b45', 'alexnet'),
    ('f27dbf2dbd5ce9a80b102d89c7483342cd33cb31', 'densenet121'),
    ('b6c8a95717e3e761bd88d145f4d0a214aaa515dc', 'densenet161'),
    ('2603f878403c6aa5a71a124c4a3307143d6820e9', 'densenet169'),
    ('1cdbc116bc3a1b65832b18cf53e1cb8e7da017eb', 'densenet201'),
    ('a5050dbcbfc54908fc1b7700698eb7ffbc150417', 'inceptionv3'),
    ('f0046a3da4e150b85da58ea31913dbb86e7540d1', 'mobilenet0.25'),
    ('0130d2aa2f7e3a63db35579b243e25f512172371', 'mobilenet0.5'),
    ('84c801e27b1ac5040b6e945c0d1f8d49e9a59f3d', 'mobilenet0.75'),
    ('efbb2ca3881998d5a8c5cb6c457a28c1e085fed5', 'mobilenet1.0'),
    ('36da4ff1867abccd32b29592d79fc753bca5a215', 'mobilenetv2_1.0'),
    ('e2be7b72a79fe4a750d1dd415afedf01c3ea818d', 'mobilenetv2_0.75'),
    ('aabd26cd335379fcb72ae6c8fac45a70eab11785', 'mobilenetv2_0.5'),
    ('ae8f9392789b04822cbb1d98c27283fc5f8aa0a7', 'mobilenetv2_0.25'),
    ('a0666292f0a30ff61f857b0b66efc0228eb6a54b', 'resnet18_v1'),
    ('48216ba99a8b1005d75c0f3a0c422301a0473233', 'resnet34_v1'),
    ('cc729d95031ca98cf2ff362eb57dee4d9994e4b2', 'resnet50_v1'),
    ('d988c13d6159779e907140a638c56f229634cb02', 'resnet101_v1'),
    ('acfd09703b113143af9c33898bad8b6154fd6fb0', 'resnet152_v1'),
    ('a81db45fd7b7a2d12ab97cd88ef0a5ac48b8f657', 'resnet18_v2'),
    ('9d6b80bbc35169de6b6edecffdd6047c56fdd322', 'resnet34_v2'),
    ('ecdde35339c1aadbec4f547857078e734a76fb49', 'resnet50_v2'),
    ('18e93e4f48947e002547f50eabbcc9c83e516aa6', 'resnet101_v2'),
    ('f2695542de38cf7e71ed58f02893d82bb409415e', 'resnet152_v2'),
    ('264ba4970a0cc87a4f15c96e25246a1307caf523', 'squeezenet1.0'),
    ('33ba0f93753c83d86e1eb397f38a667eaf2e9376', 'squeezenet1.1'),
    ('dd221b160977f36a53f464cb54648d227c707a05', 'vgg11'),
    ('ee79a8098a91fbe05b7a973fed2017a6117723a8', 'vgg11_bn'),
    ('6bc5de58a05a5e2e7f493e2d75a580d83efde38c', 'vgg13'),
    ('7d97a06c3c7a1aecc88b6e7385c2b373a249e95e', 'vgg13_bn'),
    ('e660d4569ccb679ec68f1fd3cce07a387252a90a', 'vgg16'),
    ('7f01cf050d357127a73826045c245041b0df7363', 'vgg16_bn'),
    ('ad2f660d101905472b83590b59708b71ea22b2e5', 'vgg19'),
    ('f360b758e856f1074a85abd5fd873ed1d98297c3', 'vgg19_bn'),
    ('4fa2e1ad96b8c8d1ba9e5a43556cd909d70b3985', 'vgg16_atrous'),
    ('0e169fbb64efdee6985c3c175ec4298c4bda0298', 'ssd_300_vgg16_atrous_voc'),
    ('daf8181b615b480236fcb8474545077891276945', 'ssd_512_vgg16_atrous_voc'),
    ('9c8b225a552614e4284a0f647331bfdc6940eb4a', 'ssd_512_resnet50_v1_voc'),
    ('2cc0f93edf1467f428018cc7261d3246dfa15259', 'ssd_512_resnet101_v2_voc'),
    ('37c180765a4eb3e67751d6bacac47bb9156f5fff', 'ssd_512_mobilenet1.0_voc'),
    ('447328d89d70ae1e2ca49226b8d834e5a5456df3', 'faster_rcnn_resnet50_v1b_voc'),
    ('b302ad8a8660345c368448141d8acf30b5a3801d', 'ssd_300_vgg16_atrous_coco'),
    ('5c86064290c05eccbdd88475376c71c595c8325c', 'ssd_512_vgg16_atrous_coco'),
    ('c48351620d4f0cbc49e4f7a84c8e67ef8fdc6e09', 'ssd_512_resnet50_v1_coco'),
    ('da9756faa5b9b4e34dedcf83ee0733d5895796ad', 'ssd_512_mobilenet1.0_coco'),
    ('5b4690fb7c5b62c44fb36c67d0642b633697f1bb', 'faster_rcnn_resnet50_v1b_coco'),
    ('a465eca35e78aba6ebdf99bf52031a447e501063', 'faster_rcnn_resnet101_v1d_coco'),
    ('a3527fdc2cee5b1f32a61e5fd7cda8fb673e86e5', 'mask_rcnn_resnet50_v1b_coco'),
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
    ('0ecdba34691be172036ddf244ff1b2eade75ffde', 'resnet50_v1b'),
    ('a455932aa95cb7dcfa05fd040b9b5a5660733c39', 'resnet101_v1b'),
    ('a5a61ee1ce5ab7c09720775b223360f3c60e211d', 'resnet152_v1b'),
    ('2a4e070854db538595cc7ee02e1a914bdd49ca02', 'resnet50_v1c'),
    ('064858f23f9878bfbbe378a88ccb25d612b149a1', 'resnet101_v1c'),
    ('75babab699e1c93f5da3c1ce4fd0092d1075f9a0', 'resnet152_v1c'),
    ('117a384ecf61490eb31ea147eb0e61e6d2b8a449', 'resnet50_v1d'),
    ('1b2b825feff86b0354642a4ab59f9b6e35e47338', 'resnet101_v1d'),
    ('cddbc86ff24a5544f57242ded0acb14ef1fbd437', 'resnet152_v1d'),
    ('25a187fa281ddc98afbcd0cc0f0646885b874b80', 'resnet50_v1s'),
    ('bd93a83c05f709a803b1221aeff0b028e6eebb03', 'resnet101_v1s'),
    ('cf74621d988ad06c6c6aa44f5597e5b600a966cc', 'resnet152_v1s'),
    ('766cdf9cc3e5b980b141643f054db5b48863f634', 'fcn_resnet101_coco'),
    ('12c2b9b3be7d4e133e52477150a9b3e616626a82', 'fcn_resnet101_voc'),
    ('3479525af7bdbf345e74e150aaae2e48174c0c5f', 'fcn_resnet50_ade'),
    ('d544440a35586f662ed1a5405ab9aa89cd750558', 'fcn_resnet101_ade'),
    ('ed817f76086abb4c3404af62ec1b5487c67642b7', 'deeplab_resnet101_coco'),
    ('311ed22c63f3ac28b5f1e1663c458f26600e62da', 'deeplab_resnet101_voc'),
    ('83247aaeeded6988790cd93ac93151514094846f', 'deeplab_resnet152_coco'),
    ('d35bea8817935d1ab310ef1e6dd06bb18c2d5f0d', 'deeplab_resnet152_voc'),
    ('c7789b237adc7253405bee57c84d53b15db45942', 'deeplab_resnet50_ade'),
    ('bf1584dfcec12063eff3075ee643e181c0f6d443', 'deeplab_resnet101_ade'),
    ('09e79ac5b6832724e82b57ec714f08205225a559', 'psp_resnet101_coco'),
    ('3f1cce66eb3942fdcc0cca68b56a5bb73c464e01', 'psp_resnet101_voc'),
    ('0c42cb735aebbffc010ca5770e3d5880995da021', 'psp_resnet50_ade'),
    ('eaaa87eb1b27c36c935b372779214bf164ad5b19', 'psp_resnet101_ade'),
    ('0f49fb59180c4d91305b858380a4fd6eaf068b6c', 'psp_resnet101_citys'),
    ('f5ece5ce1422eeca3ce2908004e469ffdf91fd41', 'yolo3_darknet53_voc'),
    ('09767802230b45af1c27697a2dad6d1ebaacc1e2', 'yolo3_darknet53_coco'),
    ('2189ea49720a116dead245b9b252301cffa18d28', 'darknet53'),
    ('b5538ef10557243511b9b46063aa4c40790d74ba', 'senet_154'),
    ]}

apache_repo_url = 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'
_url_format = '{repo_url}gluon/models/{file_name}.zip'

def short_hash(name):
    if name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha1[name][:8]

def get_model_file(name, tag=None, root=os.path.join('~', '.mxnet', 'models')):
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
    use_tag = isinstance(tag, str)
    if use_tag:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=tag)
    else:
        file_name = '{name}-{short_hash}'.format(name=name,
                                                 short_hash=short_hash(name))
    root = os.path.expanduser(root)
    file_path = os.path.join(root, file_name+'.params')
    if use_tag:
        sha1_hash = tag
    else:
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
