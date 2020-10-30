import os
import json
import logging
import hashlib
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path

from .io.video_io import VideoSortedFolderReader, VideoFrameReader
from .utils.serialization_utils import save_json, load_json, ComplexEncoder, \
    save_pickle, load_pickle


logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class FieldNames:
    """
    Keys found in annotation dict
    """
    # Entity Fields
    TIME = 'time'
    LABELS = 'labels'
    ID = 'id'
    BLOB = 'blob'

    # Sample fields
    METADATA = 'metadata'
    ENTITY_LIST = 'entities'
    SAMPLE_FILE = 'sample_file'

    # Dataset fields
    SAMPLE_DICT = 'samples'
    SAMPLES = 'samples'
    DATASET_METADATA = 'metadata'
    DATASET_VERSION = 'version'
    CLASS_MAPPING = 'class_mapping'

    # Data fields
    DATE_ADDED = 'date_added'
    DURATION = 'duration'
    DATA_PATH = 'data_path'
    FILENAME = 'filename'
    BASE_DIR = 'base_dir'
    FILE_EXT = 'file_ext'
    FPS = 'fps'
    NUM_FRAMES = 'number_of_frames'
    RESOLUTION = 'resolution'
    WIDTH = 'width'
    HEIGHT = 'height'
    DATASET_SOURCE = 'data_source'
    SOURCE_ID = 'source_id'
    SAMPLE_SOURCE = 'sample_source'
    ORIG_ID = 'orig_id'
    TEMPORAL_SEGMENTS = 'temporal_segments'
    DESCRIPTION = 'description'
    BOUNDING_BOXES = 'bb'
    HEAD_BBOX = "bb_head"
    FACE_BBOX = "bb_face"
    MASK = 'mask'
    KEYPOINTS = 'keypoints'
    CONFIDENCE = 'confidence'
    DEFAULT_VALUES = 'default_values'
    DEFAULT_GT = 'default_gt'
    FORMAT_VERSION = 'format_version'
    DATE_MODIFIED = 'last_modified'
    CHANGES = 'changes'
    KEY_HASH = 'key_hash'
    FRAME_IDX = 'frame_idx'

    @classmethod
    def get_key_hash(cls):
        return 0


class SplitNames:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def get_instance_list(cls, raw_entity_list):
    return [cls(raw_info=x) for x in raw_entity_list]


class AnnoEntity:
    """
    One annotation entity.
    A entity can refer to a video label, or a time segment annotation, or a frame annotation
    It has a required field "time", which should be in milliseconds
    Other than that, there are a few optional fields:
        id: if the entity has an id
        labels: if the entity has some categorical labels
        blob: other annotation information of this entity
    """

    def __init__(self, time=None, id=None, raw_info=None, validate=True):
        self._raw_info = raw_info
        self._time = None
        self._labels = None
        self._mask = None
        self._blob = {}

        self.id = None
        self.bbox = None
        self.keypoints = None
        self.confidence = None

        if raw_info is not None:
            if time is not None or id is not None:
                log.warning("time or id were specified, but raw_info was set and so they will be ignored")
            self._parse_info()
        else:
            if validate:
                self.time = time
            else:
                self._time = time
            self.id = id

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        if isinstance(time, (int, float)):
            assert time >= 0
            self._time = time
        elif isinstance(time, str):
            self._time = float(time)
        elif isinstance(time, tuple):
            assert len(time) == 2
            self._time = float(time[0]), float(time[1])

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, new_labels):
        if self._labels:
            # compare keys and report warning
            diff_keys = set(self._labels.keys()).difference(new_labels.keys())
            log.warning("updating the labels with different keys: {}".format(diff_keys))
        self._labels = new_labels


    @property
    def keypoints_xyv(self):
        return np.vstack([np.asarray(self.keypoints[0::3], dtype=np.float32),
                np.asarray(self.keypoints[1::3], dtype=np.float32),
                np.asarray(self.keypoints[2::3], dtype=np.float32)]).T

    @keypoints_xyv.setter
    def keypoints_xyv(self, kp_xyv):
        self.keypoints = list(np.asarray(kp_xyv).flatten())

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    @property
    def frame_num(self):
        if FieldNames.FRAME_IDX not in self.blob:
            return None
        return self.blob[FieldNames.FRAME_IDX]

    @frame_num.setter
    def frame_num(self, frame_num):
        self.blob[FieldNames.FRAME_IDX] = frame_num

    @property
    def blob(self):
        return self._blob

    @blob.setter
    def blob(self, new_blob):
        if self._blob:
            # compare keys and report warning
            diff_keys = set(self._blob.keys()).difference(new_blob.keys())
            log.warning("updating the blob with different keys: {}".format(diff_keys))
        self._blob = new_blob

    def to_dict(self):
        FN = FieldNames
        out = dict()

        out[FN.TIME] = self.time

        if self.id is not None:
            out[FN.ID] = self.id

        if self.labels is not None:
            out[FN.LABELS] = self.labels

        if self.confidence is not None:
            out[FN.CONFIDENCE] = self.confidence

        if self.bbox is not None:
            out[FN.BOUNDING_BOXES] = self.bbox

        if self.mask is not None:
            out[FN.MASK] = self.mask

        if self.keypoints is not None:
            out[FN.KEYPOINTS] = self.keypoints

        if self.blob:
            out[FN.BLOB] = self.blob

        return out

    def _parse_info(self):
        FN = FieldNames

        raw_info = self._raw_info

        self._time = raw_info[FN.TIME]

        self.id = raw_info.get(FN.ID, None)
        self.bbox = raw_info.get(FN.BOUNDING_BOXES, None)
        self.keypoints = raw_info.get(FN.KEYPOINTS, None)
        self.confidence = raw_info.get(FN.CONFIDENCE, 1.0)
        self._labels = raw_info.get(FN.LABELS, None)
        self._mask = raw_info.get(FN.MASK, None)
        self._blob = raw_info.get(FN.BLOB, {})

    ### helper functions ###
    @property
    def x(self):
        return self.bbox[0] if self.bbox else None
    @property
    def y(self):
        return self.bbox[1] if self.bbox else None
    @property
    def w(self):
        return self.bbox[2] if self.bbox else None
    @property
    def h(self):
        return self.bbox[3] if self.bbox else None


class DataReader:
    def __init__(self, data_sample, max_frame_deviation=1, fps=None):

        self._data_sample = data_sample
        self._data_path = data_sample.data_path
        self._frame_reader = None
        self._frame_iter = None

        if fps is None:
            fps = data_sample.fps
        self._fps = fps

        self._max_frame_deviation = max_frame_deviation
        self._period = 1.0 / fps * 1000
        self._time_diff_cutoff = int(self._max_frame_deviation * self._period) - 1

        self._frame_reader = self._data_sample.frame_reader

    def __iter__(self):
        self._frame_iter = iter(self._frame_reader)
        return self

    def __next__(self):
        frame, ts = next(self._frame_iter)
        if frame is None:
            raise StopIteration
        entities = self._data_sample.get_entities_near_time(int(ts), self._time_diff_cutoff)
        return frame, ts, entities

    def __getitem__(self, item):
        frame, ts = self._frame_reader[item]
        entities = self._data_sample.get_entities_near_time(int(ts), self._time_diff_cutoff)
        return frame, ts, entities


class DataSample:
    """
    One sample in the dataset. This can be a video file or an image file.
    It contains a list of entities as annotations.
    Each data sample can have some meta data. For example in videos, one can have
        FPS
        Duration
        Number of frames
        Source id
        ...
    """
    _NO_SERIALIZE_FIELDS = ("_dataset",)

    def __init__(self, id, raw_info=None, root_path=None, metadata=None, dataset=None):
        self._raw_info = raw_info
        self._id = id
        self._entities = []
        if metadata is not None:
            self._metadata = dict(metadata)
        else:
            self._metadata = {}
        self._dataset = dataset
        self._filepath = None
        self._lazy_loaded = False
        self._root_path = None
        self._data_root_path = None
        self._cache_root_path = None
        self._raw_entities = None
        self._time_entity_dict = None
        self._entity_times = None
        self._times_unsorted = True
        self._id_entity_dict = None
        self._frame_entity_dict = None
        self._label_entity_dict = None
        self._init_entity_fields()

        self.set_root_path(root_path)
        if self._raw_info:
            self._parse()

    def set_root_path(self, root_path):
        self._root_path = root_path
        if root_path:
            self._data_root_path = GluonCVMotionDataset.get_data_path_from_root(root_path)
            self._cache_root_path = GluonCVMotionDataset.get_cache_path_from_root(root_path)

    def _set_filepath(self, filepath, already_loaded=False):
        self._filepath = filepath
        self._lazy_loaded = already_loaded

    @property
    def id(self):
        return self._id

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, new_md):
        if self._metadata:
            # compare keys and report warning
            diff_keys = set(self._metadata.keys()).difference(new_md.keys())
            log.warning("updating the metadata with different keys: {}".format(diff_keys))
        self._metadata = new_md

    def _lazy_init(self):
        if self._filepath and not self._lazy_loaded:
            self._lazy_load()
        if (self._raw_entities is not None) and not self._entities:
            self._entities = get_instance_list(AnnoEntity, self._raw_entities)
            self._raw_entities = None
            self._init_entity_fields()

    def __len__(self):
        return self.metadata[FieldNames.NUM_FRAMES]

    @property
    def duration(self):
        return self.metadata[FieldNames.DURATION]

    @property
    def entities(self) -> [AnnoEntity]:
        self._lazy_init()
        return self._entities

    @property
    def data_relative_path(self):
        if FieldNames.DATA_PATH in self.metadata:
            data_path = self.metadata[FieldNames.DATA_PATH]
        if FieldNames.BASE_DIR in self.metadata:
            data_path = self.metadata[FieldNames.BASE_DIR]
        return data_path

    @data_relative_path.setter
    def data_relative_path(self, data_path):
        self.metadata[FieldNames.DATA_PATH] = data_path

    @property
    def data_path(self):
        data_path = self.data_relative_path
        if self._data_root_path:
            data_path = os.path.join(self._data_root_path, data_path)

        return data_path

    @property
    def frame_reader(self):
        data_path = self.data_path
        if os.path.isdir(data_path):
            frame_reader = VideoSortedFolderReader(data_path, self.fps)
        else:
            frame_reader = VideoFrameReader(data_path)
        return frame_reader

    def get_cache_file(self, cache_name, extension=''):
        rel_path = os.path.splitext(self.data_relative_path)[0] if os.path.isfile(self.data_path) else self.data_relative_path
        return os.path.join(self._cache_root_path, cache_name, rel_path + extension)

    @property
    def fps(self):
        fps = self.metadata.get(FieldNames.FPS, None)
        return fps

    @property
    def period(self):
        """Retrieves the period in milliseconds (1000 / fps), if fps is unset, returns None"""
        fps = self.fps
        return 1000 / fps if fps else None

    @property
    def width(self):
        width = self.metadata.get(FieldNames.RESOLUTION, {}).get(FieldNames.WIDTH, None)
        return width

    @property
    def height(self):
        height = self.metadata.get(FieldNames.RESOLUTION, {}).get(FieldNames.HEIGHT, None)
        return height

    @property
    def num_minutes(self):
        frames = self.metadata[FieldNames.NUM_FRAMES]
        fps = self.fps if self.fps > 1 else 30.
        return frames/fps/60.

    def get_data_reader(self):
        return DataReader(self)

    def _init_entity_fields(self):
        self._time_entity_dict = defaultdict(list)
        self._id_entity_dict = defaultdict(list)
        self._frame_entity_dict = defaultdict(list)
        self._label_entity_dict = defaultdict(list)
        self._entity_times = []
        for entity in self.entities:
            self._update_key_dicts(entity)

    @property
    def frame_num_entity_dict(self):
        self._lazy_init()
        return self._frame_entity_dict

    def get_entities_for_frame_num(self, frame_idx):
        return self.frame_num_entity_dict[frame_idx]

    @property
    def id_entity_dict(self):
        self._lazy_init()
        return self._id_entity_dict

    def get_entities_with_id(self, id):
        return self.id_entity_dict[id]

    @property
    def time_entity_dict(self):
        self._lazy_init()
        return self._time_entity_dict

    def get_entities_at_time(self, time):
        return self.time_entity_dict[time]

    @property
    def label_entity_dict(self):
        self._lazy_init()
        return self._label_entity_dict

    def get_entities_with_label(self, label):
        return self.label_entity_dict[label]

    def _get_entity_times_sorted(self):
        if self._times_unsorted:
            self._entity_times.sort()
            self._times_unsorted = False
        return self._entity_times

    def get_entities_near_time(self, time, time_diff_cutoff=None):
        import bisect
        self._lazy_init()

        if time_diff_cutoff is None:
            # Set to the period msec - 1
            time_diff_cutoff = int(1.0 / self.fps * 1000) - 1

        entity_times = self._get_entity_times_sorted()

        if not entity_times:
            return []

        insert_pos = bisect.bisect(entity_times, time)
        time_left = entity_times[max(insert_pos - 1, 0)]
        time_right = entity_times[min(insert_pos, len(entity_times)-1)]

        diff_left = abs(time_left - time)
        diff_right = abs(time_right - time)

        if diff_right < diff_left:
            closest_time = time_right
            diff = diff_right
        else:
            closest_time = time_left
            diff = diff_left

        if diff < time_diff_cutoff:
            entities = self._time_entity_dict[closest_time]
        else:
            entities = []

        return entities

    def _update_key_dicts(self, entity):
        self._id_entity_dict[entity.id].append(entity)
        new_time = entity.time not in self._time_entity_dict
        self._time_entity_dict[entity.time].append(entity)
        if entity.time is not None and new_time:
            self._entity_times.append(entity.time)
            self._times_unsorted = True
        # an entity could have multiple labels
        if entity.labels is not None:
            for k, v in entity.labels.items():
                self._label_entity_dict[k].append(entity)
        if entity.frame_num is not None:
            self._frame_entity_dict[entity.frame_num].append(entity)
        else:
            if entity.time is not None and self.fps:
                # Time in seconds * fps = frame_num
                frame_num = round((entity.time / 1000) * self.fps)
            else:
                frame_num = None
            self._frame_entity_dict[frame_num].append(entity)

    def add_entity(self, entities):
        """
        Add a new entity or a list of entities to the sample
        :param entities:
        :return:
        """
        self._lazy_init()

        if isinstance(entities, AnnoEntity):
            self._entities.append(entities)
            self._update_key_dicts(entities)
        else:
            self._entities.extend(entities)
            for entity in entities:
                self._update_key_dicts(entity)

    def get_copy_without_entities(self, new_id=None):
        """
        :return: A new DataSample with the same id and metadata but no entities
        """
        if new_id is None:
            new_id = self.id
        return DataSample(new_id, root_path=self._root_path, metadata=self.metadata, dataset=self._dataset)

    def filter_entities(self, filter_fn):
        """
        :param filter_fn: When true, keep the entity, otherwise omit it
        """
        new_sample = self.get_copy_without_entities()
        for entity in self.entities:
            if filter_fn(entity):
                new_sample.add_entity(entity)
        return new_sample

    def get_non_empty_frames(self, filter_fn=None, fps=0):
        """
          Return indexes of all valid frames with the specified fps,
          whose annotation exists
        """
        if fps == 0:
            fps = self.fps
        interval = int(np.ceil(self.fps / fps))

        frame_idxs = []
        for idx in range(0, len(self), interval):
            entities = self.get_entities_for_frame_num(idx)
            if filter_fn is not None:
                entities, _ = filter_fn(entities)
            if len(entities) > 0:
                frame_idxs.append(idx)
        return sorted(frame_idxs)

    def to_dict(self, include_id=False, lazy_load_format=False):
        """
        Dump the information in this sample to a dict
        :return:
        """
        out = dict()

        if lazy_load_format and self._filepath:
            if self._metadata:
                out[FieldNames.METADATA] = self._metadata
            out[FieldNames.SAMPLE_FILE] = self._filepath
            return out

        self._lazy_init()

        if include_id:
            out[FieldNames.ID] = self.id
        out[FieldNames.METADATA] = self.metadata
        out[FieldNames.ENTITY_LIST] = [x.to_dict() for x in self.entities]
        return out

    def dump(self, filename, indent=0, include_id=True, **kwargs):
        save_json(self.to_dict(include_id=include_id), filename, indent=indent, **kwargs)

    def _get_lazy_load_path(self):
        if not self._filepath or not self._dataset:
            raise ValueError("Cannot get lazy load path without a filepath and dataset")
        base_path = Path(self._dataset.anno_path).with_suffix("")
        if self._filepath is True:
            filepath = base_path / (self.id + ".json")
        else:
            filepath = base_path / self._filepath
        return filepath

    def _lazy_load(self):
        filepath = self._get_lazy_load_path()
        self._raw_info = load_json(filepath)
        if self.metadata and self._raw_info.get(FieldNames.METADATA) != self.metadata:
            log.info("metadata did not match lazy loaded value for sample: {}, ignoring loaded value, this should be"
                      " resolved next time you dump the dataset".format(self.id))
            self._raw_info[FieldNames.METADATA] = self.metadata
        self._parse()
        self._lazy_loaded = True

    def _dump_for_lazy_load(self):
        if not self._lazy_loaded:
            log.debug("nothing lazy loaded to dump, returning")
            return
        filepath = self._get_lazy_load_path()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.dump(filepath, include_id=False)

    def clear_lazy_loaded(self, clear_metadata=False):
        if not self._lazy_loaded:
            log.debug("nothing lazy loaded so nothing to clear")
            return
        self._entities = []
        self._raw_info = None
        self._init_entity_fields()
        if clear_metadata:
            self._metadata = {}
        self._lazy_loaded = False

    @classmethod
    def load(cls, filename, **kwargs):
        data = load_json(filename)
        raw_info = data["raw_info"] if "raw_info" in data else data
        return cls(data[FieldNames.ID], raw_info=raw_info, **kwargs)

    def _parse(self):
        self._metadata = self._raw_info.get(FieldNames.METADATA, {})
        # Lazy load entities for speed when loading dataset
        self._raw_entities = self._raw_info.get(FieldNames.ENTITY_LIST, [])
        filepath = self._raw_info.get(FieldNames.SAMPLE_FILE)
        if filepath is not None:
            self._filepath = filepath

    def copy(self):
        new = pickle.loads(pickle.dumps(self))
        for f in self._NO_SERIALIZE_FIELDS:
            setattr(new, f, getattr(self, f))
        return new

    def __getstate__(self):
        # Used by pickle and deepcopy, this prevents trying to copy the whole dataset due to the dataset back reference
        return {k: v for k, v in vars(self).items() if k not in self._NO_SERIALIZE_FIELDS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for f in self._NO_SERIALIZE_FIELDS:
            setattr(self, f, None)

    def __enter__(self):
        self._lazy_init()
        return self

    def __exit__(self):
        self.clear_lazy_loaded()


class GluonCVMotionDataset:
    ANNO_DIR = "annotation"
    CACHE_DIR = "cache"
    DATA_DIR = "raw_data"

    _DEFAULT_ANNO_FILE = "anno.json"
    _DEFAULT_SPLIT_FILE = "splits.json"

    def __init__(self, annotation_file=None, root_path=None, split_file=None, load_anno=True):
        """
        GluonCVMotionDataset
        :param annotation_file: The path to the annotation file, either a full path or a path relative to the root
         annotation path (root_path/annotation/), defaults to 'anno.json'
        :param root_path: The root path of the dataset, containing the 'annotation', 'cache', and 'raw_data' folders.
         If left empty it will be inferred from the annotation_file path by searching up until the 'annotation' folder
         is found, then going one more level up
        :param split_file: The path to the split file relative to the annotation file. It will be relative to the root
         annotation path instead if it starts with './'
        :param load_anno: Whether to load the annotation file, will cause an exception if it is true and file does not
         exist. Set this to false if you are just trying to write a new annotation file for example in an ingestion
         script
        """

        # a dict of DataSample instances
        import indexed
        self._samples = indexed.IndexedOrderedDict()
        self._splits = {}
        self._metadata = {}

        if annotation_file is None:
            annotation_file = self._DEFAULT_ANNO_FILE
            log.info("Annotation file not provided, defaulting to '{}'".format(annotation_file))

        self._root_path = self._get_root_path(root_path, annotation_file)

        if self._root_path:
            if not os.path.isdir(self._root_path):
                raise ValueError("Expected root folder but was not found at: {}".format(self._root_path))

            self._anno_path = os.path.join(self._root_path, self.ANNO_DIR, annotation_file)
            self._data_path = self.get_data_path_from_root(self._root_path)
            self._cache_path = self.get_cache_path_from_root(self._root_path)

            if not os.path.isdir(self._data_path):
                raise ValueError("Expected data folder but was not found at: {}".format(self._data_path))
        else:
            log.warning('Root path was not set for dataset, this should only happen when loading a lone annotation'
                        ' file for inspection')
            self._anno_path = annotation_file
            self._data_path = None
            self._cache_path = None

        if load_anno:
            if os.path.exists(self._anno_path):
                log.info('Loading annotation file {}...'.format(self._anno_path))
                # load annotation file
                if self._get_pickle_path().exists():
                    log.info('Found pickle file, loading this instead')
                loaded_pickle = self._load_pickle()
                if not loaded_pickle:
                    self._parse_anno(self._anno_path)
                self._split_path = self._get_split_path(split_file, self._anno_path)
                self._load_split()
            else:
                raise ValueError(
                    "load_anno is true but the anno path does not exist at: {}".format(self._anno_path))
        else:
            log.info('Skipping loading for annotation file {}'.format(self._anno_path))
            self._split_path = self._get_split_path(split_file, self._anno_path)

    def __len__(self):
        return len(self._samples)

    def __contains__(self, item):
        return item in self._samples

    def __getitem__(self, item):
        return self._samples[item]

    def __iter__(self):
        for item in self._samples.items():
            yield item

    @classmethod
    def get_data_path_from_root(cls, root_path):
        return os.path.join(root_path, cls.DATA_DIR)

    @classmethod
    def get_cache_path_from_root(cls, root_path):
        return os.path.join(root_path, cls.CACHE_DIR)

    def _get_root_path(self, root_path, annotation_file):
        if root_path is None:
            dirpath = os.path.dirname(annotation_file)
            if self.ANNO_DIR in dirpath:
                while os.path.basename(dirpath) != self.ANNO_DIR and dirpath != '/':
                    dirpath = os.path.dirname(dirpath)
                root_path = os.path.abspath(os.path.dirname(dirpath))
                log.info("Dataset root path inferred to be: {}".format(root_path))
        return root_path

    def _get_split_path(self, split_file, anno_path):
        split_path = split_file
        if split_path is None:
            split_path = self._DEFAULT_SPLIT_FILE
        if not os.path.isabs(split_path):
            if split_path.startswith("./"):
                anno_dir = os.path.join(self._root_path, self.ANNO_DIR)
            else:
                anno_dir = os.path.dirname(anno_path)
            split_path = os.path.join(anno_dir, split_path)
            split_subpath = split_path.replace(self._root_path or "", "").lstrip(os.path.sep)
            log.info("Split subpath: {}".format(split_subpath))
        return split_path

    @property
    def iter_samples(self):
        """
        returns a iterator of samples
        :return:
        """
        return self._samples.items()

    @property
    def samples(self):
        return self._samples.items()

    @property
    def sample_ids(self):
        return self._samples.keys()

    @property
    def sample_values(self):
        return self._samples.values()

    def get_split_ids(self, splits=None):
        if splits is None:
            # Default is return all ids
            return self._samples.keys()
        if isinstance(splits, str):
            splits = [splits]

        all_ids = []
        for split in splits:
            if split not in self._splits:
                log.warning("Provided split: {} was not in dataset".format(split))
            else:
                all_ids.extend(self._splits[split])

        return all_ids

    def get_split_samples(self, splits=None):
        split_ids = self.get_split_ids(splits)
        samples = []
        for split_id in split_ids:
            if split_id in self._samples:
                samples.append((split_id, self._samples[split_id]))
            else:
                log.info(f"Dataset is missing sample: {split_id} in split {self.get_use_for_id(split_id)}, skipping")
        return samples

    @property
    def train_samples(self):
        return self.get_split_samples(SplitNames.TRAIN)

    @property
    def val_samples(self):
        return self.get_split_samples(SplitNames.VAL)

    @property
    def trainval_samples(self):
        return self.get_split_samples([SplitNames.TRAIN, SplitNames.VAL])

    @property
    def test_samples(self):
        return self.get_split_samples(SplitNames.TEST)

    @property
    def all_samples(self):
        samples = self.get_split_samples([SplitNames.TRAIN, SplitNames.VAL, SplitNames.TEST])
        if not len(samples):
            samples = self.samples
        return samples

    def get_use_for_id(self, id):
        for use in self._splits:
            if id in self._splits[use]:
                return use
        return None

    @property
    def version(self):
        return __version__

    @property
    def metadata(self):
        return self._metadata

    @property
    def name(self):
        if not self._root_path:
            return None
        return os.path.basename(self._root_path)

    @property
    def root_path(self):
        return self._root_path

    @property
    def anno_root_path(self):
        return os.path.join(self._root_path, self.ANNO_DIR)

    @property
    def cache_root_path(self):
        return self._cache_path

    @property
    def data_root_path(self):
        return self._data_path

    @property
    def anno_path(self):
        return self._anno_path

    def _get_anno_subpath(self, anno_path, with_ext):
        subpath = Path(anno_path).relative_to(self.anno_root_path)
        if not with_ext:
            subpath = subpath.with_suffix("")
        return str(subpath)

    def get_anno_subpath(self, with_ext=False):
        return self._get_anno_subpath(self._anno_path, with_ext)

    def get_anno_suffix(self):
        subpath = self.get_anno_subpath()
        return "_" + subpath.replace(os.sep, "_")

    @property
    def split_path(self):
        return self._split_path

    def get_split_subpath(self, with_ext=False):
        return self._get_anno_subpath(self._split_path, with_ext)

    def get_split_suffix(self):
        subpath = self.get_split_subpath()
        return "_" + subpath.replace(os.sep, "_")

    @metadata.setter
    def metadata(self, new_md):
        if self._metadata:
            # compare keys and report warning
            diff_keys = set(self._metadata.keys()).difference(new_md.keys())
            log.warning("updating the metadata with different keys: {}".format(diff_keys))
        self._metadata = new_md

    @property
    def description(self):
        return self._metadata.get(FieldNames.DESCRIPTION, "")

    @description.setter
    def description(self, description):
        self._metadata[FieldNames.DESCRIPTION] = description

    def add_sample(self, sample:DataSample, dump_directly=False):
        # create a new sample so it functions just as it would if it were loaded from disk
        new_sample = DataSample(sample.id, raw_info=sample.to_dict(), root_path=self.root_path, dataset=self)
        self._samples[sample.id] = new_sample
        if dump_directly:
            new_sample._set_filepath(True, already_loaded=True)
            new_sample._dump_for_lazy_load()
            new_sample.clear_lazy_loaded()
        return new_sample

    def dumps(self, encoder=ComplexEncoder, **kwargs):
        return json.dumps(self._to_dict(), cls=encoder, **kwargs)

    def dump(self, filename=None, indent=0, **kwargs):
        if filename is None:
            filename = self._anno_path
            anno_dir = os.path.dirname(self._anno_path)
            if not os.path.exists(anno_dir):
                try:
                    os.mkdir(anno_dir)
                except OSError:
                    pass
        save_json(self._to_dict(), filename, indent=indent, **kwargs)

    def _get_pickle_path(self):
        return Path(self._anno_path).with_suffix(".pkl")

    def _anno_mod_time(self):
        return Path(self._anno_path).stat().st_mtime

    def dump_pickle(self, filepath=None, **kwargs):
        if filepath is None:
            filepath = str(self._get_pickle_path())
        modified_time = self._anno_mod_time()
        to_pickle = {
            "_samples": self._samples,
            "_metadata": self._metadata,
            "_raw_info": self._raw_info,
            "modified_time": modified_time
        }
        save_pickle(to_pickle, filepath, **kwargs)

    def _load_pickle(self, filepath=None):
        import datetime
        if filepath is None:
            filepath = str(self._get_pickle_path())
        if not os.path.exists(filepath):
            return False
        log.info('Loading pickle file {}'.format(filepath))
        try:
            loaded_dict = load_pickle(filepath)
        except OSError:
            log.warning("Failed to load pickle")
            return False

        stored_time = loaded_dict["modified_time"]
        modified_time = self._anno_mod_time()
        if stored_time == modified_time:
            self._samples = loaded_dict["_samples"]
            self._metadata = loaded_dict["_metadata"]
            self._raw_info = loaded_dict["_raw_info"]
            return True
        else:
            log.info(('The pickle stored modification time did not match the annotation, so not loading,'
                     ' please remove and regenerate the pickle, renaming to .old').format(filepath))
            new_filepath = str(filepath) + ".old_" + str(datetime.datetime.now()).replace(" ", "_")
            try:
                Path(filepath).rename(new_filepath)
            except OSError:
                pass
            return False

    def _parse_anno(self, annotation_file):
        json_info = load_json(annotation_file)
        self._raw_info = json_info
        log.info("loaded anno json")

        # load metadata
        assert FieldNames.DATASET_METADATA in self._raw_info, \
            "key: {} should present in the annotation file, we only got {}".format(
                FieldNames.DATASET_METADATA, self._raw_info.keys())
        self._metadata.update(self._raw_info[FieldNames.DATASET_METADATA])

        # check key map hash
        key_hash = self._metadata.get(FieldNames.KEY_HASH, '')
        assert key_hash == FieldNames.get_key_hash(), "Key list not matching. " \
                                                      "Maybe this annoation file is created with other versions." \
                                                      "Current version is {}".format(self.version)

        # load samples
        sample_dict = self._raw_info.get(FieldNames.SAMPLE_DICT, dict())
        root_path = self.root_path
        for k in sorted(sample_dict.keys()):
            self._samples[k] = DataSample(k, sample_dict[k], root_path=root_path, dataset=self)
            # self._samples[k] = sample_dict[k]
        log.info("loaded {} samples".format(len(self._samples)))

    def _load_split(self):
        if not os.path.exists(self._split_path):
            log.warning("Split path {} not found, skipping loading".format(self._split_path))
            return
        self._splits = load_json(self._split_path)
        split_sample_nums = {k: len(v) for k, v in self._splits.items()}
        log.info("Loaded splits with # samples: {}".format(split_sample_nums))

    @property
    def splits(self):
        return dict(self._splits)

    @splits.setter
    def splits(self, split_dict):
        self._splits = split_dict

    def dump_splits(self, filename=None, indent=2):
        if filename is None:
            filename = self._split_path
        save_json(self._splits, filename, indent=indent)

    def _to_dict(self, dump_sample_files=True):
        # add the version information to metadata
        self._metadata[FieldNames.DATASET_VERSION] = self.version
        self._metadata[FieldNames.KEY_HASH] = FieldNames.get_key_hash()

        dump_dict = dict()
        dump_dict[FieldNames.DATASET_METADATA] = self._metadata
        all_samples_dict = {}
        for sample_id, sample in self._samples.items():
            sample_dict = sample.to_dict(lazy_load_format=True)
            if sample._lazy_loaded and dump_sample_files:
                sample._dump_for_lazy_load()
            all_samples_dict[sample_id] = sample_dict
        dump_dict[FieldNames.SAMPLE_DICT] = all_samples_dict
        return dump_dict


def get_resized_256_video_location(data_sample: DataSample) -> str:
    return get_resized_video_location(data_sample, 256)


def get_resized_video_location(data_sample: DataSample, short_edge_res:int) -> str:
    return data_sample.get_cache_file('rgb_{}_mp4'.format(short_edge_res), extension='.mp4')


def get_vis_video_location(data_sample: DataSample) -> str:
    return data_sample.get_cache_file('full_res_mp4', extension='.mp4')


def get_vis_thumb_location(data_sample: DataSample) -> str:
    return data_sample.get_cache_file('thumbnails', extension='.jpg')


def get_vis_gt_location(data_sample: DataSample, cache_subpath: str) -> str:
    return data_sample.get_cache_file(os.path.join('gt_vis_json', cache_subpath), extension='.json')


def get_gt_data_sample_location(data_sample: DataSample) -> str:
    return data_sample.get_cache_file('gt_data_sample_json', extension='.json')
