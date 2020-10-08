"""Dataset implementation for specific task(s)"""
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

try:
    import mxnet as mx
    MXDataset = mx.gluon.data.Dataset
except ImportError:
    MXDataset = object

logger = logging.getLogger()

def _absolute_pathify(df, root=None, column='image'):
    """Convert relative paths to absolute"""
    if root is None:
        return df
    assert column in df.columns
    assert isinstance(root, str), 'Invalid root path: {}'.format(root)
    root = os.path.abspath(os.path.expanduser(root))
    for i, row in df.iterrows():
        path = df.at[i, 'image']
        if not os.path.isabs(path):
            df.at[i, 'image'] = os.path.join(root, os.path.expanduser(path))
    return df


class ImageClassificationDataset(pd.DataFrame):
    # preserved properties that will be copied to a new instance
    _metadata = ['classes', 'to_mxnet']

    def __init__(self, data, classes=None, **kwargs):
        if isinstance(data, str) and data.endswith('csv'):
            self = self.from_csv(data, root=kwargs.get('root', None))
        self.classes = classes
        super().__init__(data, **kwargs)

    @property
    def _constructor(self):
        return ImageClassificationDataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    @classmethod
    def from_csv(cls, csv_file, root=None):
        df = pd.read_csv(csv_file)
        assert 'image' in df.columns, "`image` column is required, used for accessing the original images"
        if not 'class' in df.columns:
            logger.debug('class not in columns, no access to labels of images')
            classes = None
        else:
            classes = df.class.unique()
        df = _absolute_pathify(df, root=root, column='image')
        return cls(df, classes=classes)

    @classmethod
    def from_folder(cls, root, exts=('.jpg', '.jpeg', '.png')):
        """A dataset for loading image files stored in a folder structure.
        like::
            root/car/0001.jpg
            root/car/xxxa.jpg
            root/car/yyyb.jpg
            root/bus/123.png
            root/bus/023.jpg
            root/bus/wwww.jpg
        """
        synsets = []
        items = {'image': [], 'class': []}
        assert isinstance(root, str)
        root = os.path.abspath(os.path.expanduser(root))

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                logger.debug('Ignoring %s, which is not a directory.'%path, stacklevel=3)
                continue
            label = len(synsets)
            synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in exts:
                    logger.debug('Ignoring %s of type %s. Only support %s'%(
                        filename, ext, ', '.join(exts)))
                    continue
                items['image'].append(filename)
                items['class'].append(label)
        return cls(items, classes=synsets)

    @classmethod
    def from_path_func(cls, fn):
        # create from a function
        raise NotImplementedError

    @classmethod
    def from_label_func(cls, fn):
        # create from a label function
        raise NotImplementedError


class ObjectDetectionDataset(pd.DataFrame):
    # preserved properties that will be copied to a new instance
    _metadata = ['dataset_type', 'classes', 'pack', 'unpack', 'is_packed', 'to_mxnet']

    def __init__(self, data, dataset_type=None, classes=None, **kwargs):
        # dataset_type will be used to determine metrics, if None then auto resolve at runtime
        self.dataset_type = dataset_type
        # if classes is not specified(None), then infer from the annotations
        self.classes = classes
        super().__init__(data, **kwargs)

    @property
    def _constructor(self):
        return ObjectDetectionDataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    @classmethod
    def from_voc(cls, root, splits=None, exts=('.jpg', '.jpeg', '.png')):
        # construct from pascal VOC forma
        from ...data.pascal_voc.detection import CustomVOCDetectionBase
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
                    logger.warn('Invalid bbox: {%s} for {%s}', str(xml_box), anno_file.name)
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
    def from_coco(cls, path):
        # construct from COCO format
        raise NotImplementedError

    @classmethod
    def from_path_func(cls, fn):
        # create from a function
        raise NotImplementedError

    @classmethod
    def from_label_func(cls, fn):
        # create from a label function
        raise NotImplementedError

    def pack(self):
        """Convert object-centric entries to image-centric entries.
        Where multiple entries belonging to single image can be merged to rois column.
        """
        if self.is_packed():
            return self
        rois_columns = ['class', 'xmin', 'ymin', 'xmax', 'ymax', 'difficult']
        image_attr_columns = ['width', 'height']
        new_df = self.groupby(['image'], as_index=False).agg(list).reset_index(drop=True)
        new_df['rois'] = new_df.agg(
            lambda y : [{k : y[new_df.columns.get_loc(k)][i] for k in rois_columns if k in new_df.columns} for i in range(len(y[new_df.columns.get_loc('class')]))], axis=1)
        new_df = new_df.drop(rois_columns, axis=1, errors='ignore')
        new_df['image_attr'] = new_df.agg(
            lambda y: {k : y[new_df.columns.get_loc(k)][0] for k in image_attr_columns if k in new_df.columns}, axis=1)
        new_df = new_df.drop(image_attr_columns, axis=1, errors='ignore')
        return self.__class__(new_df.reset_index(drop=True))

    def unpack(self):
        if not self.is_packed():
            return self
        new_df = self.explode('rois')
        new_df = pd.concat([new_df.drop(['rois'], axis=1), new_df['rois'].apply(pd.Series)], axis=1)
        new_df = pd.concat([new_df.drop(['image_attr'], axis=1), new_df['image_attr'].apply(pd.Series)], axis=1)
        return self.__class__(new_df.reset_index(drop=True))

    def is_packed(self):
        return 'rois' in self.columns and 'xmin' not in self.columns

    def to_mxnet(self):
        """Return a mxnet based iterator that returns ndarray and labels"""
        return _MXObjectDetectionDataset(self)


class _MXObjectDetectionDataset(MXDataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ObjectDetectionDataset
        DataFrame as ObjectDetectionDataset.

    """
    def __init__(self, dataset):
        assert isinstance(dataset, ObjectDetectionDataset)
        if not dataset.is_packed():
            dataset = dataset.pack()
        assert 'image' in dataset.columns
        assert 'rois' in dataset.columns
        assert 'image_attr' in dataset.columns
        self._dataset = dataset
        self.classes = self._dataset.classes
        import mxnet as mx
        self._imread = mx.image.imread

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        rois = self._dataset['rois'][idx]
        img_attr = self._dataset['image_attr'][idx]
        width = img_attr['width']
        height = img_attr['height']
        img = self._imread(im_path)
        def convert_entry(roi):
            return [float(roi[key]) for key in ['xmin', 'ymin', 'xmax', 'ymax']] + \
                [self.classes.index(roi['class']), float(roi['difficult'])]
        label = np.array([convert_entry(roi) for roi in rois])
        label[:, (0, 2)] *= width
        label[:, (1, 3)] *= height
        return img, label
