"""Visual Tracker Benchmark.
Code adapted from https://github.com/STVIR/pysot"""
import json
import os
from glob import glob
from tqdm import tqdm
from mxnet.gluon.data import dataset
from gluoncv.utils.filesystem import try_import_cv2


class Video(object):
    """
    Abstract video class. get video class information for example imgs.

    Parameters
    ----------
        name : str
            video name
        root: str
            dataset root
        video_dir: str
            video directory
        init_rect: list
            init rectangle
        img_names: str
            image names
        gt_rect: list
            groundtruth rectangle
        attr: str
            attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
                 gt_rect, attr, load_img=False):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}
        self.img_names = [os.path.join(root, x) for x in img_names]
        self.imgs = None
        cv2 = try_import_cv2()

        if load_img:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]
        else:
            img = cv2.imread(self.img_names[0])
            assert img is not None, self.img_names[0]
            self.width = img.shape[1]
            self.height = img.shape[0]

    def load_img(self):
        if self.imgs is None:
            cv2 = try_import_cv2()
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]

    def free_img(self):
        self.imgs = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            cv2 = try_import_cv2()
            return cv2.imread(self.img_names[idx]), self.gt_traj[idx]
        else:
            return self.imgs[idx], self.gt_traj[idx]

    def __iter__(self):
        for i in range(len(self.img_names)):
            if self.imgs is not None:
                yield self.imgs[i], self.gt_traj[i]
            else:
                cv2 = try_import_cv2()
                yield cv2.imread(self.img_names[i]), self.gt_traj[i]

class OTBVideo(Video):
    """
    OTBVideo class. Including video operation

    Parameters
    ----------
        name : str
            video name
        root: str
            dataset root
        video_dir: str
            video directory
        init_rect: list
            init rectangle
        img_names: str
            image names
        gt_rect: list
            groundtruth rectangle
        attr: str
            attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
                 gt_rect, attr, load_img=False):
        super(OTBVideo, self).__init__(name, root, video_dir,
                                       init_rect, img_names, gt_rect, attr, load_img)

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        open txt and load_tracker
        Parameters
        ----------
            path : str
                path to result
            tracker_name : list
                name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                             if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if not os.path.exists(traj_file):
                if self.name == 'FleetFace':
                    txt_name = 'fleetface.txt'
                elif self.name == 'Jogging-1':
                    txt_name = 'jogging_1.txt'
                elif self.name == 'Jogging-2':
                    txt_name = 'jogging_2.txt'
                elif self.name == 'Skating2-1':
                    txt_name = 'skating2_1.txt'
                elif self.name == 'Skating2-2':
                    txt_name = 'skating2_2.txt'
                elif self.name == 'FaceOcc1':
                    txt_name = 'faceocc1.txt'
                elif self.name == 'FaceOcc2':
                    txt_name = 'faceocc2.txt'
                elif self.name == 'Human4-2':
                    txt_name = 'human4_2.txt'
                else:
                    txt_name = self.name[0].lower()+self.name[1:]+'.txt'
                traj_file = os.path.join(path, name, txt_name)
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f:
                    pred_traj = [list(map(float, x.strip().split(',')))
                                 for x in f.readlines()]
                    if len(pred_traj) != len(self.gt_traj):
                        print(name, len(pred_traj), len(self.gt_traj), self.name)
                    if store:
                        self.pred_trajs[name] = pred_traj
                    else:
                        return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())
        return None

class OTBTracking(dataset.Dataset):
    """OTB Visual Tracker Benchmark.

    Parameters
    ----------
    name : str
        name to data, and name to dataset json Default is 'OTB2015'
    dataset_root: str
        path to dataset root
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(OTBTracking, self).__init__()
        self.name = name
        self.dataset_root = dataset_root
        with open(os.path.join(self.dataset_root, self.name+'.json'), 'r') as f:
            meta_data = json.load(f)
        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+self.name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = OTBVideo(video,
                                          self.dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'],
                                          load_img)
        # set attr
        attr = []
        for x in self.videos.values():
            attr += x.attr
        attr = set(attr)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
        for x in attr:
            self.attr[x] = []
        for k, v in self.videos.items():
            for attr_ in v.attr:
                self.attr[attr_].append(k)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.videos[idx]
        elif isinstance(idx, int):
            return self.videos[sorted(list(self.videos.keys()))[idx]]
        return None

    def __len__(self):
        return len(self.videos)

    def set_tracker(self, path, tracker_names):
        """
        Args:
            path: path to tracker results,
            tracker_names: list of tracker name
        """
        self.tracker_path = path
        self.tracker_names = tracker_names
