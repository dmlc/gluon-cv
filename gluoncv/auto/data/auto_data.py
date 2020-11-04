"""Auto data preparation."""
import os
import re
from pathlib import Path
import yaml
from ...utils.download import download
from ...utils.filesystem import unzip, untar, PathTree

def url_data(url, path=None, overwrite=False, overwrite_folder=False, sha1_hash=None, root=None, disp_depth=1):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        ~/.gluoncv directory with same name as in url.
        You can also change the default behavior by editing ~/.gluoncv/config.yaml.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    overwrite_folder : bool, optional
        Whether to extract file to destination folder if already exists. You may use this option if you suspect
        the destination is corrupted or some files are missing.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    root : str, optional
        Destination root dir to store extracted files. By default it's default in ~/.gluoncv directory.
    disp_depth : int, optional
        If set > 0, will print out the tree structure of extracted dataset folder with maximum `disp_depth`.

    Returns
    -------
    str or tuple of str
        The file path of the downloaded file.
    """
    fname = Path(path or URLs.path(url, c_key='archive'))
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname = download(url, path=str(fname.resolve()), overwrite=overwrite, sha1_hash=sha1_hash)
    extract_root = URLs.path(url, c_key='data')
    extract_root = extract_root.parent.joinpath(extract_root.stem)
    extract_root.mkdir(parents=True, exist_ok=True)
    if fname.endswith('.zip'):
        folder = unzip(fname, root=root if root else extract_root, strict=overwrite_folder)
    elif fname.endswith('gz'):
        folder = untar(fname, root=root if root else extract_root, strict=overwrite_folder)
    else:
        raise ValueError('Unknown url data with file: {}'.format(fname))

    if disp_depth > 0:
        path_tree = PathTree(folder, disp_depth)
        print(path_tree)

    return Path(folder)


class Config:
    "Setup config at `~/.gluoncv` unless it exists already."
    config_path = Path(os.getenv('MXNET_HOME', '~/.gluoncv')).expanduser()
    config_file = config_path/'config.yml'

    def __init__(self):
        self.config_path.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            self.create_config()
        self.d = self.load_config()

    def __getitem__(self, k):
        k = k.lower()
        if k not in self.d:
            k = k + '_path'
        return Path(self.d[k])

    def __getattr__(self, k):
        if k == 'd':
            raise AttributeError
        return self[k]

    def __setitem__(self, k, v):
        self.d[k] = str(v)
    def __contains__(self, k):
        return k in self.d

    def load_config(self):
        "load and return config if version equals 2 in existing, else create new config."
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            if 'version' in config and config['version'] == 2:
                return config
            elif 'version' in config:
                self.create_config(config)
            else:
                self.create_config()
        return self.load_config()

    def create_config(self, cfg=None):
        "create new config with default paths and set `version` to 2."
        config = {'data_path':    str(self.config_path/'datasets'),
                  'archive_path': str(self.config_path/'archive'),
                  'storage_path': '/tmp',
                  'model_path':   str(self.config_path/'models'),
                  'version':      2}
        if cfg is not None:
            cfg['version'] = 2
            config = merge(config, cfg)
        self.save_file(config)

    def save(self):
        self.save_file(self.d)

    def save_file(self, config):
        "save config file at default config location `~/.gluoncv/config.yml`."
        with self.config_file.open('w') as f:
            yaml.dump(config, f, default_flow_style=False)


# pylint: disable=bad-whitespace
class URLs():
    "Global constants for dataset and model URLs."
    LOCAL_PATH = Path.cwd()
    MDL = 'http://files.fast.ai/models/'
    S3  = 'https://s3.amazonaws.com/fast-ai-'
    URL = f'{S3}sample/'

    S3_IMAGE    = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_COCO     = f'{S3}coco/'

    # main datasets
    ADULT_SAMPLE        = f'{URL}adult_sample.tgz'
    BIWI_SAMPLE         = f'{URL}biwi_sample.tgz'
    CIFAR               = f'{URL}cifar10.tgz'
    COCO_SAMPLE         = f'{S3_COCO}coco_sample.tgz'
    COCO_TINY           = f'{S3_COCO}coco_tiny.tgz'
    HUMAN_NUMBERS       = f'{URL}human_numbers.tgz'
    # IMDB                = f'{S3_NLP}imdb.tgz'
    IMDB_SAMPLE         = f'{URL}imdb_sample.tgz'
    ML_SAMPLE           = f'{URL}movie_lens_sample.tgz'
    ML_100k             = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    MNIST_SAMPLE        = f'{URL}mnist_sample.tgz'
    MNIST_TINY          = f'{URL}mnist_tiny.tgz'
    MNIST_VAR_SIZE_TINY = f'{S3_IMAGE}mnist_var_size_tiny.tgz'
    PLANET_SAMPLE       = f'{URL}planet_sample.tgz'
    PLANET_TINY         = f'{URL}planet_tiny.tgz'
    IMAGENETTE          = f'{S3_IMAGE}imagenette2.tgz'
    IMAGENETTE_160      = f'{S3_IMAGE}imagenette2-160.tgz'
    IMAGENETTE_320      = f'{S3_IMAGE}imagenette2-320.tgz'
    IMAGEWOOF           = f'{S3_IMAGE}imagewoof2.tgz'
    IMAGEWOOF_160       = f'{S3_IMAGE}imagewoof2-160.tgz'
    IMAGEWOOF_320       = f'{S3_IMAGE}imagewoof2-320.tgz'
    IMAGEWANG           = f'{S3_IMAGE}imagewang.tgz'
    IMAGEWANG_160       = f'{S3_IMAGE}imagewang-160.tgz'
    IMAGEWANG_320       = f'{S3_IMAGE}imagewang-320.tgz'

    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats.tgz'

    # image classification datasets
    CALTECH_101  = f'{S3_IMAGE}caltech_101.tgz'
    CARS         = f'{S3_IMAGE}stanford-cars.tgz'
    CIFAR_100    = f'{S3_IMAGE}cifar100.tgz'
    CUB_200_2011 = f'{S3_IMAGE}CUB_200_2011.tgz'
    FLOWERS      = f'{S3_IMAGE}oxford-102-flowers.tgz'
    FOOD         = f'{S3_IMAGE}food-101.tgz'
    MNIST        = f'{S3_IMAGE}mnist_png.tgz'
    PETS         = f'{S3_IMAGE}oxford-iiit-pet.tgz'

    # Image localization datasets
    BIWI_HEAD_POSE     = f"{S3_IMAGELOC}biwi_head_pose.tgz"
    CAMVID             = f'{S3_IMAGELOC}camvid.tgz'
    CAMVID_TINY        = f'{URL}camvid_tiny.tgz'
    LSUN_BEDROOMS      = f'{S3_IMAGE}bedroom.tgz'
    PASCAL_2007        = f'{S3_IMAGELOC}pascal_2007.tgz'
    PASCAL_2012        = f'{S3_IMAGELOC}pascal_2012.tgz'

    # Medical Imaging datasets
    #SKIN_LESION        = f'{S3_IMAGELOC}skin_lesion.tgz'
    SIIM_SMALL         = f'{S3_IMAGELOC}siim_small.tgz'

    @staticmethod
    def path(url='.', c_key='archive'):
        "Return local path where to download based on `c_key`"
        fname = url.split('/')[-1]
        local_path = URLs.LOCAL_PATH / ('models' if c_key == 'models' else 'datasets')/fname
        if local_path.exists():
            return local_path
        return Config()[c_key]/fname

_URL_REGEX = re.compile(
    r'^(?:http|ftp)s?://' # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    r'localhost|' #localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?' # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def is_url(url_like):
    if not isinstance(url_like, str):
        return False
    return re.match(_URL_REGEX, url_like) is not None
