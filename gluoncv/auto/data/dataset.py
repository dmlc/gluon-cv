"""Dataset implementation for specific task(s)"""
# pylint: disable=consider-using-generator
import logging
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from ..data import is_url, url_data
from ...data.mscoco.utils import try_import_pycocotools
from ...utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
try:
    import mxnet as mx
    MXDataset = mx.gluon.data.Dataset
except ImportError:
    MXDataset = object
    mx = None

logger = logging.getLogger()

def _absolute_pathify(df, root=None, column='image'):
    """Convert relative paths to absolute"""
    if root is None:
        return df
    assert column in df.columns
    assert isinstance(root, str), 'Invalid root path: {}'.format(root)
    root = os.path.abspath(os.path.expanduser(root))
    for i, _ in df.iterrows():
        path = df.at[i, 'image']
        if not os.path.isabs(path):
            df.at[i, 'image'] = os.path.join(root, os.path.expanduser(path))
    return df


class ImageClassificationDataset(pd.DataFrame):
    """ImageClassification dataset as DataFrame.

    Parameters
    ----------
    data : the input for pd.DataFrame
        The input data.
    classes : list of str, optional
        The class synsets for this dataset, if `None`, it will infer from the data.

    """
    # preserved properties that will be copied to a new instance
    _metadata = ['classes', 'to_mxnet', 'show_images', 'random_split']

    def __init__(self, data, classes=None, **kwargs):
        root = kwargs.pop('root', None)
        if isinstance(data, str) and data.endswith('csv'):
            data = self.from_csv(data, root=root)
        self.classes = classes
        super().__init__(data, **kwargs)

    @property
    def _constructor(self):
        return ImageClassificationDataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def random_split(self, test_size=0.1, val_size=0, random_state=None):
        r"""Randomly split the dataset into train/val/test sets.
        Note that it's perfectly fine to set `test_size` or `val_size` to 0, where the
        returned splits will be empty dataframes.

        Parameters
        ----------
        test_size : float
            The ratio for test set, can be in range [0, 1].
        val_size : float
            The ratio for validation set, can be in range [0, 1].
        random_state : int, optional
            If not `None`, will set the random state of numpy.random engine.

        Returns
        -------
        train, val, test - (DataFrame, DataFrame, DataFrame)
            The returned dataframes for train/val/test

        """
        assert 0 <= test_size < 1.0
        assert 0 <= val_size < 1.0
        assert (val_size + test_size) < 1.0, 'val_size + test_size is larger than 1.0!'
        if random_state:
            np.random.seed(random_state)
        test_mask = np.random.rand(len(self)) < test_size
        test = self[test_mask]
        trainval = self[~test_mask]
        val_mask = np.random.rand(len(trainval)) < val_size
        val = trainval[val_mask]
        train = trainval[~val_mask]
        return train, val, test

    def show_images(self, indices=None, nsample=16, ncol=4, shuffle=True, resize=224, fontsize=20):
        r"""Display images in dataset.

        Parameters
        ----------
        indices : iterable of int, optional
            The image indices to be displayed, if `None`, will generate `nsample` indices.
            If `shuffle` == `True`(default), the indices are random numbers.
        nsample : int, optional
            The number of samples to be displayed.
        ncol : int, optional
            The column size of ploted image matrix.
        shuffle : bool, optional
            If `shuffle` is False, will always sample from the begining.
        resize : int, optional
            The image will be resized to (resize, resize) for better visual experience.
        fontsize : int, optional
            The fontsize for the title
        """
        if indices is None:
            if not shuffle:
                indices = range(nsample)
            else:
                indices = list(range(len(self)))
                np.random.shuffle(indices)
                indices = indices[:min(nsample, len(indices))]
        images = [cv2.cvtColor(cv2.resize(cv2.imread(self.at[idx, 'image']), (resize, resize), \
            interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for idx in indices if idx < len(self)]
        titles = None
        if 'label' in self.columns:
            titles = [self.classes[int(self.at[idx, 'label'])] + ': ' + str(self.at[idx, 'label']) \
                for idx in indices if idx < len(self)]
        _show_images(images, cols=ncol, titles=titles, fontsize=fontsize)

    def to_mxnet(self):
        """Return a mxnet based iterator that returns ndarray and labels"""
        return _MXImageClassificationDataset(self)

    @classmethod
    def from_csv(cls, csv_file, root=None):
        r"""Create from csv file.

        Parameters
        ----------
        csv_file : str
            The path for csv file.
        root : str
            The relative root for image paths stored in csv file.

        """
        if is_url(csv_file):
            csv_file = url_data(csv_file, disp_depth=0)
        df = pd.read_csv(csv_file)
        assert 'image' in df.columns, "`image` column is required, used for accessing the original images"
        if not 'label' in df.columns:
            logger.info('label not in columns, no access to labels of images')
            classes = None
        else:
            classes = df['label'].unique()
        df = _absolute_pathify(df, root=root, column='image')
        return cls(df, classes=classes)

    @classmethod
    def from_folder(cls, root, exts=('.jpg', '.jpeg', '.png')):
        r"""A dataset for loading image files stored in a folder structure.
        like::
            root/car/0001.jpg
            root/car/xxxa.jpg
            root/car/yyyb.jpg
            root/bus/123.png
            root/bus/023.jpg
            root/bus/wwww.jpg

        Parameters
        -----------
        root : str or pathlib.Path
            The root folder
        exts : iterable of str
            The image file extensions
        """
        if is_url(root):
            root = url_data(root)
        synsets = []
        items = {'image': [], 'label': []}
        if isinstance(root, Path):
            assert root.exists(), '{} not exist'.format(str(root))
            root = str(root.resolve())
        assert isinstance(root, str)
        root = os.path.abspath(os.path.expanduser(root))

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                logger.debug('Ignoring %s, which is not a directory.', path, stacklevel=3)
                continue
            label = len(synsets)
            synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in exts:
                    logger.debug('Ignoring %s of type %s. Only support %s',
                                 filename, ext, ', '.join(exts))
                    continue
                items['image'].append(filename)
                items['label'].append(label)
        return cls(items, classes=synsets)

    @classmethod
    def from_folders(cls, root, train='train', val='val', test='test', exts=('.jpg', '.jpeg', '.png')):
        """Method for loading splited datasets under root.
        like::
            root/train/car/0001.jpg
            root/train/car/xxxa.jpg
            root/train/car/yyyb.jpg
            root/val/bus/123.png
            root/test/bus/023.jpg
            root/test/bus/wwww.jpg
        will be loaded into three splits, with 3/1/2 images, respectively.
        You can specify the sub-folder names of `train`/`val`/`test` individually. If one particular sub-folder is not
        found, the corresponding returned dataset will be `None`.

        Example:
        >>> train_data, val_data, test_data = ImageClassificationDataset.from_folders('./data', val='validation')
        >> assert len(train_data) == 3


        Parameters
        ----------
        root : str or pathlib.Path or url
            The root dir for the entire dataset, if url is provided, the data will be downloaded and extracted.
        train : str
            The sub-folder name for training images.
        val : str
            The sub-folder name for training images.
        test : str
            The sub-folder name for training images.
        exts : iterable of str
            The supported image extensions when searching for sub-sub-directories.

        Returns
        -------
        (train_data, val_data, test_data) of type tuple(ImageClassificationDataset, )
            splited datasets, can be `None` if no sub-directory found.

        """
        if is_url(root):
            root = url_data(root)
        if isinstance(root, Path):
            assert root.exists(), '{} not exist'.format(str(root))
            root = str(root.resolve())
        assert isinstance(root, str)
        root = os.path.abspath(os.path.expanduser(root))
        train_root = os.path.join(root, train)
        val_root = os.path.join(root, val)
        test_root = os.path.join(root, test)
        empty = cls({'image': [], 'label': []})
        train_data, val_data, test_data = empty, empty, empty
        # train
        if os.path.isdir(train_root):
            train_data = cls.from_folder(train_root, exts=exts)
        else:
            raise ValueError('Train split does not exist: {}'.format(train))
        # val
        if os.path.isdir(val_root):
            val_data = cls.from_folder(val_root, exts=exts)
        # test
        if os.path.isdir(test_root):
            test_data = cls.from_folder(test_root, exts=exts)

        # check synsets, val/test synsets can be subsets(order matters!) or exact matches of train synset
        if len(val_data) and not _check_synsets(train_data.classes, val_data.classes):
            warnings.warn('Train/val synsets does not match: {} vs {}'.format(train_data.classes, val_data.classes))
        if len(test_data) and not _check_synsets(train_data.classes, test_data.classes):
            warnings.warn('Train/val synsets does not match: {} vs {}'.format(train_data.classes, test_data.classes))

        return train_data, val_data, test_data

    @classmethod
    def from_name_func(cls, im_list, fn, root=None):
        """Short summary.

        Parameters
        ----------
        cls : type
            Description of parameter `cls`.
        im_list : type
            Description of parameter `im_list`.
        fn : type
            Description of parameter `fn`.
        root : type
            Description of parameter `root`.

        Returns
        -------
        type
            Description of returned object.

        """
        # create from a function parsed from name
        synsets = []
        items = {'image': [], 'label': []}
        for im in im_list:
            if isinstance(im, Path):
                path = str(im.resolve())
            else:
                assert isinstance(im, str)
                if root is not None and not os.path.isabs(im):
                    path = os.path.abspath(os.path.join(root, os.path.expanduser(im)))
            items['image'].append(path)
            label = fn(Path(path))
            if isinstance(label, (int, bool, str)):
                label = str(label)
            else:
                raise ValueError('Expect returned label to be (str, int, bool), received {}'.format(type(label)))
            if label not in synsets:
                synsets.append(label)
            items['label'].append(synsets.index(label))  # int label id
        return cls(items, classes=synsets)

    @classmethod
    def from_name_re(cls, im_list, fn, root=None):
        # create from a re parsed from name
        raise NotImplementedError

    @classmethod
    def from_label_func(cls, label_list, fn):
        # create from a function parsed from labels
        raise NotImplementedError


class _MXImageClassificationDataset(MXDataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ImageClassificationDataset
        DataFrame as ImageClassificationDataset.

    """
    def __init__(self, dataset):
        if mx is None:
            raise RuntimeError('Unable to import mxnet which is required.')
        assert isinstance(dataset, ImageClassificationDataset)
        assert 'image' in dataset.columns
        self._has_label = 'label' in dataset.columns
        self._dataset = dataset
        self.classes = self._dataset.classes
        self._imread = mx.image.imread

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        img = self._imread(im_path)
        label = None
        if self._has_label:
            label = self._dataset['label'][idx]
        return img, label


class ObjectDetectionDataset(pd.DataFrame):
    """ObjectDetection dataset as DataFrame.

    Parameters
    ----------
    data : the input for pd.DataFrame
        The input data.
    dataset_type : str, optional
        The dataset type, can be voc/coco or more, used to annotate the optional fields.
    classes : list of str, optional
        The class synsets for this dataset, if `None`, it will infer from the data.

    """
    # preserved properties that will be copied to a new instance
    _metadata = ['dataset_type', 'classes', 'pack', 'unpack', 'is_packed',
                 'to_mxnet', 'color_map', 'show_images', 'random_split']

    def __init__(self, data, dataset_type=None, classes=None, **kwargs):
        # dataset_type will be used to determine metrics, if None then auto resolve at runtime
        self.dataset_type = dataset_type
        # if classes is not specified(None), then infer from the annotations
        self.classes = classes
        self.color_map = {}
        super().__init__(data, **kwargs)

    @property
    def _constructor(self):
        return ObjectDetectionDataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    @classmethod
    def from_voc(cls, root, splits=None, exts=('.jpg', '.jpeg', '.png')):
        """construct from pascal VOC format.
        Normally you will see a structure like:

        ├── VOC2007
        │   ├── Annotations
        │   ├── ImageSets
        |   |   ├── Main
        |   |   |   ├── train.txt
        |   |   |   ├── test.txt
        │   ├── JPEGImages

        Parameters
        ----------
        root : str or url
            The root directory for VOC, e.g., the `VOC2007`. If an url is provided, it will be downloaded and extracted.
        splits : tuple of str, optional
            If given, will search for this name in `ImageSets/Main/`, e.g., ('train', 'test')
        exts : tuple of str, optional
            The supported image formats.

        """
        if is_url(root):
            root = url_data(root)
        rpath = Path(root).expanduser()
        img_list = []
        class_names = set()
        if splits:
            logger.debug('Use splits: %s for root: %s', str(splits), root)
            if isinstance(splits, str):
                splits = [splits]
            for split in splits:
                split_file = rpath / 'ImageSets' / 'Main' / split
                if not split_file.resolve().exists():
                    split_file = rpath / 'ImageSets' / 'Main' / (split + '.txt')
                if not split_file.resolve().exists():
                    raise FileNotFoundError(split_file)
                with split_file.open(mode='r') as fi:
                    img_list += [line.split()[0].strip() for line in fi.readlines()]
        else:
            logger.debug('No split provided, use full image list in %s, with extension %s',
                         str(rpath/'JPEGImages'), str(exts))
            if not isinstance(exts, (list, tuple)):
                exts = [exts]
            for ext in exts:
                img_list.extend([rp.stem for rp in rpath.glob('JPEGImages/*' + ext)])
        d = {'image': [], 'rois': [], 'image_attr': []}
        for stem in img_list:
            basename = stem + '.xml'
            anno_file = (rpath / 'Annotations' / basename).resolve()
            tree = ET.parse(anno_file)
            xml_root = tree.getroot()
            size = xml_root.find('size')
            im_path = xml_root.find('filename').text
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            rois = []
            for obj in xml_root.iter('object'):
                try:
                    difficult = int(obj.find('difficult').text)
                except ValueError:
                    difficult = 0
                cls_name = obj.find('name').text.strip().lower()
                xml_box = obj.find('bndbox')
                xmin = max(0, float(xml_box.find('xmin').text) - 1) / width
                ymin = max(0, float(xml_box.find('ymin').text) - 1) / height
                xmax = min(width, float(xml_box.find('xmax').text) - 1) / width
                ymax = min(height, float(xml_box.find('ymax').text) - 1) / height
                if xmin >= xmax or ymin >= ymax:
                    logger.warning('Invalid bbox: {%s} for {%s}', str(xml_box), anno_file.name)
                else:
                    rois.append({'class': cls_name,
                                 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                                 'difficult': difficult})
                    class_names.update((cls_name,))
            if rois:
                d['image'].append(str(rpath / 'JPEGImages' / im_path))
                d['rois'].append(rois)
                d['image_attr'].append({'width': width, 'height': height})
        df = pd.DataFrame(d)
        return cls(df.sort_values('image').reset_index(drop=True), dataset_type='voc', classes=list(class_names))

    @classmethod
    def from_coco(cls, anno_file, root=None, min_object_area=0, use_crowd=False):
        """Load dataset from coco format annotations.

        The structure of a default coco 2017 dataset looks like:
        .
        ├── annotations
        |   |── instances_val2017.json
        ├── train2017
        └── val2017

        The default relative root folder (if set to `None`) is `anno_file/../`.
        """
        # construct from COCO format
        try_import_pycocotools()
        from pycocotools.coco import COCO
        if isinstance(anno_file, Path):
            anno_file = str(anno_file.expanduser().resolve())
        elif isinstance(anno_file, str):
            anno_file = os.path.expanduser(anno_file)
        coco = COCO(anno_file)

        if isinstance(root, Path):
            root = str(root.expanduser().resolve())
        elif isinstance(root, str):
            root = os.path.abspath(os.path.expanduser(root))
        elif root is None:
            # try to use the default coco structure
            root = os.path.join(os.path.dirname(anno_file), '..')
            logger.info('Using default root folder: %s. Specify `root=...` if you feel it is wrong...', root)
        else:
            raise valueError("Unable to parse root: {}".format(root))

        # synsets
        classes = [c['name'] for c in coco.loadCats(coco.getCatIds())]
        # load entries
        d = {'image': [], 'rois': [], 'image_attr': []}
        image_ids = sorted(coco.getImgIds())
        for entry in coco.loadImgs(image_ids):
            if 'coco_url' in entry:
                dirname, filename = entry['coco_url'].split('/')[-2:]
                abs_path = os.path.join(root, dirname, filename)
            else:
                abs_path = os.path.join(root, entry['file_name'])
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = _check_load_coco_bbox(coco, entry, min_object_area=min_object_area, use_crowd=use_crowd)
            if not label:
                continue
            d['image_attr'].append({'width': entry['width'], 'height': entry['height']})
            d['image'].append(abs_path)
            d['rois'].append(label)
        df = pd.DataFrame(d)
        return cls(df.sort_values('image').reset_index(drop=True), dataset_type='coco', classes=list(classes))

    @classmethod
    def from_label_func(cls, fn):
        """create from a label function"""
        raise NotImplementedError

    def pack(self):
        """Convert object-centric entries to image-centric entries.
        Where multiple entries belonging to single image can be merged to rois column.

        The length of returned dataframe is the number of images in the dataset.
        """
        if self.is_packed():
            return self
        orig_classes = self.classes
        rois_columns = ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'difficult']
        image_attr_columns = ['width', 'height']
        new_df = self.groupby(['image'], as_index=False).agg(list).reset_index(drop=True)
        new_df['rois'] = new_df.agg(
            lambda y: [{k : y[new_df.columns.get_loc(k)][i] for k in rois_columns if k in new_df.columns} \
                for i in range(len(y[new_df.columns.get_loc('class')]))], axis=1)
        new_df = new_df.drop(rois_columns, axis=1, errors='ignore')
        new_df['image_attr'] = new_df.agg(
            lambda y: {k : y[new_df.columns.get_loc(k)][0] for k in image_attr_columns if k in new_df.columns}, axis=1)
        new_df = new_df.drop(image_attr_columns, axis=1, errors='ignore')
        new_df = self.__class__(new_df.reset_index(drop=True))
        new_df.classes = orig_classes
        return new_df

    def unpack(self):
        """Convert image-centric entries to object-centric entries.
        Where single entry carries multiple objects, stored in `rois` column.

        The length of returned dataframe is the number of objects in the dataset.
        """
        if not self.is_packed():
            return self
        orig_classes = self.classes
        new_df = self.explode('rois')
        new_df = pd.concat([new_df.drop(['rois'], axis=1), new_df['rois'].apply(pd.Series)], axis=1)
        new_df = pd.concat([new_df.drop(['image_attr'], axis=1), new_df['image_attr'].apply(pd.Series)], axis=1)
        new_df = self.__class__(new_df.reset_index(drop=True))
        new_df.classes = orig_classes
        return new_df

    def is_packed(self):
        """Check whether the current dataframe is providing packed representation of rois.
        """
        return 'rois' in self.columns and 'xmin' not in self.columns

    def to_mxnet(self):
        """Return a mxnet based iterator that returns ndarray and labels"""
        return _MXObjectDetectionDataset(self)

    def random_split(self, test_size=0.1, val_size=0, random_state=None):
        r"""Randomly split the dataset into train/val/test sets.
        Note that it's perfectly fine to set `test_size` or `val_size` to 0, where the
        returned splits will be empty dataframes.

        Parameters
        ----------
        test_size : float
            The ratio for test set, can be in range [0, 1].
        val_size : float
            The ratio for validation set, can be in range [0, 1].
        random_state : int, optional
            If not `None`, will set the random state of numpy.random engine.

        Returns
        -------
        train, val, test - (DataFrame, DataFrame, DataFrame)
            The returned dataframes for train/val/test

        """
        assert 0 <= test_size < 1.0
        assert 0 <= val_size < 1.0
        assert (val_size + test_size) < 1.0, 'val_size + test_size is larger than 1.0!'
        if random_state:
            np.random.seed(random_state)
        test_mask = np.random.rand(len(self)) < test_size
        test = self[test_mask]
        trainval = self[~test_mask]
        val_mask = np.random.rand(len(trainval)) < val_size
        val = trainval[val_mask]
        train = trainval[~val_mask]
        return train, val, test

    def show_images(self, indices=None, nsample=16, ncol=4, shuffle=True, resize=512, fontsize=20):
        r"""Display images in dataset.

        Parameters
        ----------
        indices : iterable of int, optional
            The image indices to be displayed, if `None`, will generate `nsample` indices.
            If `shuffle` == `True`(default), the indices are random numbers.
        nsample : int, optional
            The number of samples to be displayed.
        ncol : int, optional
            The column size of ploted image matrix.
        shuffle : bool, optional
            If `shuffle` is False, will always sample from the begining.
        resize : int, optional
            The image will be resized to (resize, resize) for better visual experience.
        fontsize : int, optional, default is 20
            The fontsize for title
        """
        df = self.pack()
        if indices is None:
            if not shuffle:
                indices = range(nsample)
            else:
                indices = list(range(len(df)))
                np.random.shuffle(indices)
                indices = indices[:min(nsample, len(indices))]
        images = [cv2.cvtColor(cv2.resize(cv2.imread(df.at[idx, 'image']), (resize, resize), \
            interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for idx in indices if idx < len(df)]
        # draw bounding boxes
        assert 'rois' in df.columns
        for i, image in enumerate(images):
            height, width, _ = image.shape
            for roi in df.at[indices[i], 'rois']:
                xmin, ymin, xmax, ymax, cls_name = \
                    roi['xmin'], roi['ymin'], roi['xmax'], roi['ymax'], roi['class']
                xmin, xmax = (np.array([xmin, xmax]) * width).astype('int')
                ymin, ymax = (np.array([ymin, ymax]) * height).astype('int')
                if cls_name not in df.color_map:
                    df.color_map[cls_name] = tuple([np.random.randint(0, 255) for _ in range(3)])
                bcolor = df.color_map[cls_name]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), bcolor, 2, lineType=cv2.LINE_AA)
            images[i] = image

        # reuse the color map
        if not self.color_map:
            self.color_map = df.color_map

        titles = ['Image(' + str(idx) + ')' for idx in indices if idx < len(df)]
        _show_images(images, cols=ncol, titles=titles, fontsize=fontsize)


class _MXObjectDetectionDataset(MXDataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ObjectDetectionDataset
        DataFrame as ObjectDetectionDataset.

    """
    def __init__(self, dataset):
        if mx is None:
            raise RuntimeError('Unable to import mxnet which is required.')
        assert isinstance(dataset, ObjectDetectionDataset)
        if not dataset.is_packed():
            dataset = dataset.pack()
        assert 'image' in dataset.columns
        assert 'rois' in dataset.columns
        self._dataset = dataset
        self.classes = self._dataset.classes
        self._imread = mx.image.imread

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        rois = self._dataset['rois'][idx]
        img = self._imread(im_path)
        width, height = img.shape[1], img.shape[0]
        def convert_entry(roi):
            return [float(roi[key]) for key in ['xmin', 'ymin', 'xmax', 'ymax']] + \
                [self.classes.index(roi['class']), float(roi.get('difficult', 0))]
        label = np.array([convert_entry(roi) for roi in rois])
        label[:, (0, 2)] *= width
        label[:, (1, 3)] *= height
        return img, label

def _check_synsets(ref_synset, other_synset):
    """Check if other_synset is part of ref_synsetself.
    Not that even if other_synset is a subset, still be careful when comparing them.

    E.g., ref: ['apple', 'orange', 'melon'], other: ['apple', 'orange'] is OK
          ref: ['apple', 'orange', 'melon'], other: ['orange', 'melon'] is not!
    """
    if ref_synset == other_synset:
        return True
    if len(other_synset) < len(ref_synset):
        if ref_synset[:len(other_synset)] == other_synset:
            return True
    return False

def _show_images(images, cols=1, titles=None, fontsize=20):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images/float(cols))), cols, n + 1)
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title, fontsize=fontsize)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def _check_load_coco_bbox(coco, entry, min_object_area=0, use_crowd=False):
    """Check and load ground-truth labels"""
    entry_id = entry['id']
    # fix pycocotools _isArrayLike which don't work for str in python3
    entry_id = [entry_id] if not isinstance(entry_id, (list, tuple)) else entry_id
    ann_ids = coco.getAnnIds(imgIds=entry_id, iscrowd=None)
    objs = coco.loadAnns(ann_ids)
    # check valid bboxes
    valid_objs = []
    width = entry['width']
    height = entry['height']
    for obj in objs:
        if obj['area'] < min_object_area:
            continue
        if obj.get('ignore', 0) == 1:
            continue
        is_crowd = obj.get('iscrowd', 0)
        if not use_crowd and is_crowd:
            continue
        # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
        xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
        # require non-zero box area
        if obj['area'] > 0 and xmax > xmin and ymax > ymin:
            cname = coco.loadCats(obj['category_id'])[0]['name']
            valid_objs.append({'xmin': xmin / width, 'ymin': ymin / height, 'xmax': xmax / width,
                               'ymax': ymax, 'class': cname, 'is_crowd': is_crowd})
    return valid_objs
