"""Simple memory/disk cached dataset for heavy duty data pre-processing tasks."""
import os
import logging
import shelve
from tqdm import tqdm

class CachedDataset(object):
    """Cached Dataset for heavy duty data-processing tasks.

    Parameters
    ----------
    dataset : callable
        Gluon dataset.
    cache_file : str
        File to store cached dataset samples.
    cleanup : bool
        Delete cached file when destroying dataset object.

    """
    def __init__(self, dataset, cache_file, cleanup=False):
        self._dataset = dataset
        self._cache_file = os.path.abspath(os.path.expanduser(cache_file))
        self._shelve = shelve.open(self._cache_file)
        self._cleanup = cleanup
        self._prepare_cache()

    def __del__(self):
        self._shelve.close()
        if self._cleanup:
            try:
                os.remove(self._cache_file)
            except OSError:
                pass

    def _prepare_cache(self):
        if set(self._shelve.keys()) == set([str(i) for i in range(len(self._dataset))]):
            logging.warn("Reuse previous cached file with {} keys...".format(len(self._dataset)))
            return
        logging.warn("Preparing cache, please be patient...")
        logging.warn("Note that all randomness will lost once cache is built.")
        for idx in tqdm(range(len(self._dataset)),
                           total=len(self._dataset),
                           unit='sample', unit_scale=False, dynamic_ncols=True):
            self._shelve[str(idx)] = self._dataset[idx]
        self._shelve.sync()

    def __getitem__(self, idx):
        return self._shelve[str(idx)]

    def __len__(self):
        return len(self._dataset)
