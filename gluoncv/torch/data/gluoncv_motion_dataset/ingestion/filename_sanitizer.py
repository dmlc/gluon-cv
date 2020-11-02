"""Avoid filename collision"""
import logging
import os
import hashlib
import re
from pathlib import Path
import tqdm

from ..dataset import GluonCVMotionDataset
from ..utils.serialization_utils import save_json


_log = logging.getLogger()
_log.setLevel(logging.DEBUG)

PATH_MAXLEN = 255
non_word_pattern = re.compile(r'[^A-Za-z0-9_\- ]+')


def hash_path(path, algorithm="blake2b", suffix_len=None, num_orig_chars=None, constant_len=False):
    # pylint: disable=missing-function-docstring
    path = Path(path)
    if suffix_len is None:
        suffix_len = len(path.suffix)
    stem = str(path.stem)
    replaced_stem = stem.replace(' ', '_')
    replaced_stem = replaced_stem.replace('-', '_')
    filtered_stem = non_word_pattern.sub('', replaced_stem)
    if len(filtered_stem) == len(stem):
        return replaced_stem

    path = str(path)
    if algorithm == "blake2b":
        # Using blake2b by default since it is fast and as good as sha-3: https://blake2.net/
        hashstr = hashlib.blake2b(path.encode(), digest_size=16).hexdigest()
    elif algorithm == "md5":
        hashstr = hashlib.md5(path.encode()).hexdigest()
    else:
        raise ValueError("Unsupported algorithm {}".format(algorithm))

    # 1 for underscore between
    max_orig_chars = PATH_MAXLEN - (len(hashstr) + 1) - suffix_len

    orig_take_chars = max_orig_chars if num_orig_chars is None \
                                     else min(num_orig_chars, max_orig_chars)
    if orig_take_chars > 0:
        trunc_stem = filtered_stem[:orig_take_chars]
        if num_orig_chars and constant_len:
            trunc_stem = trunc_stem.ljust(orig_take_chars, '_')
        new_stem = "{}_{}".format(trunc_stem, hashstr)
    else:
        new_stem = hashstr

    return new_stem


def main(anno_path, cache_name="uuid_filenames", ref_cache=None,
         ref_ext=".mp4", flatten=False, no_ext_hash=False):
    # pylint: disable=line-too-long， missing-function-docstring
    dataset = GluonCVMotionDataset(anno_path)

    id_mapping = {}
    cache_dir = Path(dataset.cache_root_path) / cache_name

    new_path_set = set()
    samples = dataset.samples
    for sample_id, sample in tqdm.tqdm(samples, mininterval=1):
        if ref_cache:
            ref_filepath = sample.get_cache_file(ref_cache, ref_ext)
        else:
            ref_filepath = sample.data_path

        orig_cache_filepath = Path(sample.get_cache_file(cache_name, Path(ref_filepath).suffix))
        orig_suffix = orig_cache_filepath.suffix
        orig_cache_rel_filepath = orig_cache_filepath.relative_to(cache_dir)
        if no_ext_hash:
            orig_cache_rel_filepath = orig_cache_rel_filepath.with_suffix("")
        orig_filename = orig_cache_rel_filepath.name

        if flatten:
            hashed_filepath = hash_path(orig_cache_rel_filepath, suffix_len=len(orig_suffix))
            uuid_rel_filepath = hashed_filepath + orig_suffix
        else:
            hashed_filename = hash_path(orig_filename, suffix_len=len(orig_suffix))
            uuid_rel_filepath = orig_cache_rel_filepath.with_name(hashed_filename + orig_suffix)
        uuid_filepath = cache_dir / uuid_rel_filepath
        if uuid_rel_filepath in new_path_set:
            existing_id = [id for id, p in id_mapping.items() if p == str(uuid_rel_filepath)]
            _log.error("Unexpected filename collision with: {} , \
                       existing id: {} , new sample id: {}".format(uuid_rel_filepath, existing_id, sample_id))
            continue
        new_path_set.add(uuid_rel_filepath)
        id_mapping[sample_id] = str(uuid_rel_filepath)
        os.makedirs(uuid_filepath.parent, exist_ok=True)
        if not os.path.exists(uuid_filepath):
            os.symlink(ref_filepath, uuid_filepath)

    save_json(id_mapping, cache_dir / "id_mapping.json")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
