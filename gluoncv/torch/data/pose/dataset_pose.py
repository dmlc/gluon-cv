"""Build dataset for pose estimation"""
# pylint: disable=line-too-long
import itertools
import copy
import logging
import numpy as np
import torch

from ..registry.catalog import DatasetCatalog, MetadataCatalog
from ..detection.detection_dataset import load_proposals_into_dataset, filter_images_with_few_keypoints, filter_images_with_only_crowd_annotations, get_detection_dataset_dicts
from ..detection.detection_dataset import print_instances_class_histogram, build_batch_data_loader, trivial_batch_collator
from ..detection.detection_utils import check_metadata_consistency, read_image, filter_empty_instances, annotations_to_instances
from ..detection.detection_utils import check_image_size, build_augmentation_ranged_clip, transform_instance_annotations
from ..detection.detection_dataset import DatasetFromList, MapDataset, DatasetMapper
from ..detection.samplers import RepeatFactorTrainingSampler, TrainingSampler, InferenceSampler
from ..transforms import instance_transforms as T
from ..transforms.instance_transforms import RandomCropWithInstance
from ..structures import BoxMode

def compute_pseudo_bbox_with_keypoint_annotation(dataset_dicts):
    """
    :param dataset_dicts:
    :return: remove person annotations without keypoints in it
    """
    for dic in dataset_dicts:
        annotation_with_pseudo_bbox = []
        for ann in dic["annotations"]:
            if "keypoints" in ann and (np.array(ann["keypoints"][2::3]) > 0).sum() > 0:
                kpt = np.array(ann["keypoints"]).reshape(17, 3)
                valid_k = kpt[kpt[:, 2] > 0]
                min_xy = valid_k.min(axis=0)
                max_xy = valid_k.max(axis=0)
                ann['bbox'] = [max(min_xy[0] - 1, 0), max(min_xy[1] - 1, 0),
                               min(max_xy[0] + 1, dic['width']) - max(min_xy[0] - 1, 0) + 1,
                               min(max_xy[1] + 1, dic['height']) - max(min_xy[1] - 1, 0) + 1]
                annotation_with_pseudo_bbox.append(ann)
        dic["annotations"] = annotation_with_pseudo_bbox
        assert(len(dic["annotations"]) > 0)

    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Compute pseudo bbox annotations with keypoints. {} images left".format(num_after)
    )
    return dataset_dicts

def get_pose_dataset_dicts(dataset_names, filter_empty=True, min_keypoints=0, compute_pseudo_bbox=True, proposal_files=None):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (list[str]): a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)
    if compute_pseudo_bbox and has_instances:
        dataset_dicts = compute_pseudo_bbox_with_keypoint_annotation(dataset_dicts)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency("thing_classes", dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass
    return dataset_dicts


def build_pose_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_pose_dataset_dicts(
        cfg.CONFIG.DATA.DATASET.TRAIN,
        filter_empty=cfg.CONFIG.DATA.DETECTION.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.CONFIG.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.CONFIG.DATA.KEYPOINT_ON
        else 0,
        compute_pseudo_bbox=cfg.CONFIG.DATA.DETECTION.COMPUTE_PSEUDO_BBOX,
        proposal_files=cfg.CONFIG.DATA.DATASET.PROPOSAL_FILES_TRAIN if cfg.CONFIG.DATA.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapperWithBasis(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.CONFIG.DATA.DETECTION.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.CONFIG.DATA.DETECTION.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.CONFIG.TRAIN.BATCH_SIZE,
        aspect_ratio_grouping=cfg.CONFIG.DATA.DETECTION.ASPECT_RATIO_GROUPING,
        num_workers=cfg.CONFIG.DATA.NUM_WORKERS,
    )

def build_pose_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper.from_config(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.CONFIG.DATA.DATASET.PROPOSAL_FILES_TEST[list(cfg.CONFIG.DATA.DATASET.VAL).index(dataset_name)]
        ]
        if cfg.CONFIG.DATA.LOAD_PROPOSALS
        else None,
    )

    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapperWithBasis(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.CONFIG.DATA.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(**DatasetMapper.from_config(cfg, is_train))
        logger = logging.getLogger(__name__)
        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation_ranged_clip(cfg, is_train)

        if cfg.CONFIG.DATA.CROP.ENABLED and is_train:
            self.augmentation.append(
                RandomCropWithInstance(
                    cfg.CONFIG.DATA.CROP.TYPE,
                    cfg.CONFIG.DATA.CROP.SIZE,
                    cfg.CONFIG.DATA.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[-1])
            )

        # fmt: off
        # TODO(zhreshold): remove this?
        self.basis_loss_on = False  # cfg.CONFIG.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set = "coco"  # cfg.CONFIG.MODEL.BASIS_MODULE.ANN_SET
        # fmt: on

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = read_image(
                dataset_dict["file_name"], format=self.image_format
            )
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        boxes = np.asarray(
            [
                BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                )
                for instance in dataset_dict["annotations"]
            ]
        )
        aug_input = T.StandardAugInput(image, boxes=boxes, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # visualize_transformed_img(np.array(utils.read_image(dataset_dict["file_name"], format=self.image_format)),
            #                           dataset_dict["annotations"], bbox_fotmat="xywh", save_name="ori_img.png")

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # visualize_transformed_img(np.array(image), annos)
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict
