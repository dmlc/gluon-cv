import logging
from pathlib import Path, PosixPath
from collections import defaultdict
import tqdm

from ..dataset import GluonCVMotionDataset, get_vis_thumb_location
from ..utils.ingestion_utils import get_chunked_id_map


_log = logging.getLogger()
_log.setLevel(logging.DEBUG)


def get_data_gen(image_files=None):
    import imagededup.utils.data_generator as data_generator
    orig_datagen = data_generator.DataGenerator
    class PathNoName(PosixPath):
        @property
        def name(self):
            return str(self)

    class DataGenerator(orig_datagen):
        def _get_image_files(self) -> None:
            self.invalid_image_idx = []
            if image_files:
                self.image_files = [PathNoName(i) for i in image_files]
            else:
                self.image_files = sorted(
                    [
                        PathNoName(i.absolute())
                        for i in self.image_dir.rglob('*.jpg')
                        if not i.name.startswith('.')]
                )  # ignore hidden files
    return DataGenerator


def run_parallel(encoder, files):
    from imagededup.utils.general_utils import parallelise
    hashes = parallelise(encoder.encode_image, files, encoder.verbose)
    hash_dict = dict(zip(files, hashes))
    return hash_dict


def set_from_dict(d):
    vals = []
    for sid, v in d.items():
        if v:
            vals.append(sid)
        for x in v:
            dup_sid = x[0]
            vals.append(dup_sid)
    return set(vals)


def select_to_keep(samples, existing_dataset=None):
    if existing_dataset is None:
        existing_dataset = set()

    def get_dim(s): return s.width * s.height
    def get_duration(s): return len(s) / s.fps * 1000
    def cond_dim(sample, keep):
        return get_dim(sample) > get_dim(keep)
    def cond_duration(sample, keep):
        return get_dim(sample) >= (get_dim(keep) * 0.95) and \
               (get_duration(sample) - get_duration(keep) > 5000)

    keep = samples[0]
    for sample in samples[1:]:
        if sample.id in existing_dataset:
            keep = sample
        elif cond_dim(sample, keep) and not cond_duration(keep, sample):
            keep = sample
        elif cond_duration(sample, keep):
            keep = sample

    remove_dups = [s for s in samples if (s != keep) and (s.id not in existing_dataset)]
    return keep, remove_dups


def get_merged_dups(duplicates):
    def get_child_dups(path, dups_set=None):
        if dups_set is None:
            dups_set = set()
        dups_set.add(path)
        dups = duplicates[path]
        for dup in dups:
            if dup not in dups_set:
                get_child_dups(dup, dups_set)
        return dups_set

    new_duplicates = {}
    for path in duplicates:
        path_dups = get_child_dups(path) - {path}
        new_duplicates[path] = [(x, -1) for x in path_dups]
    return new_duplicates


def get_unique_dups(duplicates, image_dir, dataset, thumb_to_sample_id, sample_id_to_thumb):
    selected_dups = set()
    all_dups = set()
    dups_map = defaultdict(list)
    dups_keep_lookup = {}

    for path, dups in duplicates.items():
        rel_path = str(Path(path).relative_to(image_dir))
        if rel_path not in selected_dups:
            for dup in dups:
                dup_path, score = dup
                rel_dup_path = str(Path(dup_path).relative_to(image_dir))
                all_dups.add(rel_dup_path)
                dups_map[rel_path].append((rel_dup_path, score))
            thumb_paths = [rel_path] + [str(Path(dup[0]).relative_to(image_dir)) for dup in dups
                                        if dup not in selected_dups]
            samples = [dataset[thumb_to_sample_id[p]] for p in thumb_paths]
            keep, remove_dups = select_to_keep(samples)
            remove_dups_thumbs = [sample_id_to_thumb[s.id] for s in remove_dups]
            selected_dups.update(remove_dups_thumbs)
            for dup in remove_dups_thumbs:
                dups_keep_lookup[dup] = sample_id_to_thumb[keep.id]
    return all_dups, dups_map, selected_dups


def get_files_to_inspect(dups_map, dataset, thumb_to_sample_id, orig_dataset=None, id_map=None):
    inspect_map = defaultdict(list)
    def get_samp(sid):
        sid = thumb_to_sample_id[sid]
        if orig_dataset and id_map and sid in id_map:
            return orig_dataset[id_map[sid]]
        else:
            return dataset[sid]

    for sid, v in dups_map.items():
        samp = get_samp(sid)
        for (dup_sid, score) in v:
            dup_samp = get_samp(dup_sid)
            len_diff = abs(len(samp) - len(dup_samp))
            if len_diff > 10:
                inspect_map[sid].append((dup_sid, score, len_diff))
    return inspect_map


def get_encodings(im_paths, method="cnn"):
    import imagededup.utils.data_generator as data_generator
    from imagededup.methods import PHash, CNN
    if method == "cnn":
        data_generator.DataGenerator = get_data_gen(im_paths)
        deduper = CNN()
        encodings = deduper.encode_images('/')
    elif method == "phash":
        deduper = PHash()
        encodings = run_parallel(deduper, im_paths)
    else:
        raise ValueError(f"Method {method} unknown")
    return encodings, deduper


def get_duplicates(im_paths, method="cnn", encodings=None, deduper=None, **kwargs):
    from imagededup.methods import PHash, CNN
    if encodings is None:
        encodings, deduper = get_encodings(im_paths, method)
    if method == "cnn":
        deduper = deduper or CNN()
        assert isinstance(deduper, CNN)
        thresh = "min_similarity_threshold"
        if thresh not in kwargs:
            kwargs[thresh] = 0.9
    elif method == "phash":
        deduper = deduper or PHash()
        assert isinstance(deduper, PHash)
        thresh = "max_distance_threshold"
        if thresh not in kwargs:
            kwargs[thresh] = 10
    else:
        raise ValueError(f"Method {method} unknown")
    duplicates = deduper.find_duplicates(encoding_map=encodings, scores=True, **kwargs)
    return duplicates, encodings, deduper


def main(anno_path, out_anno=None, orig_anno_file=None):
    dataset = GluonCVMotionDataset(anno_path)
    if out_anno is None:
        anno_path = Path(anno_path)
        out_anno = anno_path.with_name("{}_dedupe{}".format(anno_path.stem, anno_path.suffix))
    out_dataset = GluonCVMotionDataset(out_anno, dataset.root_path, load_anno=False)
    out_dataset.metadata = dataset.metadata
    out_dataset.description += "; deduped"
    samples = dataset.samples

    if orig_anno_file:
        orig_dataset = GluonCVMotionDataset(orig_anno_file, root_path=dataset.root_path)
        orig_id_map = get_chunked_id_map(dataset, orig_dataset, chunk_subdirs=["vid_chunks_2m_1p"],
                                         exclude_unchunked=True)
    else:
        orig_dataset = None
        orig_id_map = {}

    image_dir = Path(dataset.cache_root_path) / "thumbnails"

    thumb_paths = []
    thumb_to_sample_id = {}
    sample_id_to_thumb = {}
    for sample_id, sample in tqdm.tqdm(samples, mininterval=1):
        thumb_path = get_vis_thumb_location(sample)
        if not Path(thumb_path).exists():
            _log.warning("Missing thumbnail: {}".format(thumb_path))
            continue
        thumb_paths.append(thumb_path)
        thumb_rel_path = str(Path(thumb_path).relative_to(image_dir))
        thumb_to_sample_id[thumb_rel_path] = sample_id
        sample_id_to_thumb[sample_id] = thumb_rel_path

    def im_filter(path):
        if path.suffix.lower() not in [".jpeg", ".jpg", ".png"]:
            return False
        return True


    duplicates, _, _ = get_duplicates(thumb_paths, method="cnn", min_similarity_threshold=0.83)

    dup_set = set_from_dict(duplicates)
    addn_args = [image_dir, dataset, thumb_to_sample_id, sample_id_to_thumb]
    all_dups, dups_map, selected_dups = get_unique_dups(duplicates, *addn_args)

    print(len(selected_dups))
    dup_sample_ids = [thumb_to_sample_id[thumb_path] \
                     for thumb_path in selected_dups \
                     if "videos_MEVA" not in thumb_path \
                        and "Offline_CCTV_Footage" not in thumb_path]
    print(len(dup_sample_ids))
    for sample_id, sample in tqdm.tqdm(samples):
        if sample_id not in dup_sample_ids:
                out_dataset.add_sample(sample)

    out_dataset.dump()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
