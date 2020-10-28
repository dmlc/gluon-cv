import logging
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from multiprocessing import Pool
from subprocess import call
from timeit import default_timer as timer
from urllib.error import HTTPError
import ray


_log = logging.getLogger()
_log.setLevel(logging.DEBUG)


VIDEO_NAME_TEMPLATE = "youtube-{}.mp4"
CROPPED_VIDEO_NAME_TEMPLATE = "cropped_youtube-{}.mp4"
tmp_path = '/tmp'

@ray.remote(num_cpus=2)
def download_ray(dests, ids, num_workers):
    return download(dests, ids, num_workers, num_hosts=1)

def download(dests, ids, num_workers=1, num_hosts=1):
    import youtube_dl
    if isinstance(dests, str):
        dests = [dests]*len(ids)
    global YDL
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(tmp_path, 'youtube-%(id)s'),
        'logger': _log
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        YDL = ydl

        blocked_response = "download-blocked"
        if num_hosts > 1:
            download_blocked = True
            # Start offsets so if hosts get blocked then don't retry from beginning
            offsets = [0] * num_hosts
            step = num_hosts
            while download_blocked:
                ray.init(redis_address="localhost:6379")
                download_blocked = False
                full_offsets = [i + offset * step for i, offset in enumerate(offsets)]
                ray_results = ray.get([download_ray.remote(dests[offset::step], ids[offset::step], num_workers)
                                       for offset in full_offsets])
                _log.info("Ray job finished, processing results")
                results = []
                for i, r in enumerate(ray_results):
                    if blocked_response in r:
                        offsets[i] = r.index(blocked_response)
                        download_blocked = True
                        _log.info("Job part {} was blocked!".format(i))
                    else:
                        offsets[i] = len(r) - 1
                        _log.info("Job part {} finished".format(i))
                    results.extend(r)
                if download_blocked:
                    _log.warning("At least one download was blocked")
                    _log.info("Blocked offsets: {} ; ids per host: {}".format(offsets, len(ids) / num_hosts))
                    _log.warning("Download was blocked, sleeping and trying again")
                    ray.shutdown()
                    time.sleep(60 * 10)
        else:
            if num_workers > 1:
                pool = Pool(processes=num_workers)
                results = pool.starmap(download_single_vid, zip(dests, ids))
                pool.close()
            else:
                results = []
                for dest, id in zip(dests,ids):
                    results.append(download_single_vid(dest, id))
                    if results[-1] == blocked_response:
                        _log.warning('Found blocked request, stopping early')
                        break

        grouped_results = defaultdict(list)
        for id, result in zip(ids, results):
            grouped_results[result].append(id)
        _log.info('Results: ')
        for type in grouped_results:
            if type not in ('skipped', 'download-succeeded'):
                _log.info('{} {}: {}'.format(len(grouped_results[type]),type,', '.join(grouped_results[type])))
            else:
                _log.info('{} {}'.format(len(grouped_results[type]),type))
    return results


def launch_stress(min_cop_percent=2):
    result = subprocess.run(["pgrep", "stress"], stdout=subprocess.PIPE)
    if result.returncode:
        min_cores = math.ceil(min_cop_percent / (100 / os.cpu_count()))
        return subprocess.Popen(["nice", "-n", "19", "stress", "-c", str(min_cores)], stdout=subprocess.PIPE,
                                preexec_fn=os.setpgrp)
    return None


def stop_stress(stress_proc):
    import signal
    return os.killpg(stress_proc.pid, signal.SIGTERM)


def download_single_vid(dest:str, id:str):
    if id.find('youtube.com') != -1:
        id = id[id.find('v=')+2:]
        ndx = id.find('&')
        if ndx != -1:
            id = id[:ndx]
    elif id.find('youtu.be/'):
        id = id[id.find('youtu.be/')+9:]
    ydl = YDL
    filename = VIDEO_NAME_TEMPLATE.format(id)
    tmp_download_path = os.path.join(tmp_path, filename)
    dest_path = os.path.join(dest, filename)

    if os.path.exists(dest_path) :
        _log.info("Skipping already downloaded, id: %s", id)
        return 'skipped'

    base_filename = os.path.splitext(filename)[0]
    # The quotes are necessary to prevent expansion of wildcards like * or ? in the youtube id
    tmp_remove_path = "'{}'*".format(os.path.join(tmp_download_path, base_filename))
    def remove_tmp_files():
        _log.info("Removing temp files using path: %s", tmp_remove_path)
        call("rm {}".format(tmp_remove_path), shell=True)

    delay = random.uniform(3.0, 6.0)
    _log.info("waiting for %f sec", delay)
    time.sleep(delay)
    retries = 0
    retry_hours = [2] + [1] * 6 + [3, 10]
    max_retries = len(retry_hours)
    success = False
    total_download_attempts = 0
    while not success:
        try:
            _log.info("Starting download for id %s", id)
            start = timer()
            total_download_attempts += 1
            ydl.download(['http://www.youtube.com/watch?v=' + id])
            end = timer()
            _log.info("Downloading for id %s took %f", id, end - start)

            _log.info("Moving temp file for: %s from %s to dest of %s", id, tmp_download_path, dest_path)
            start = timer()
            ret_code = call(['mv', tmp_download_path, dest_path])
            end = timer()
            _log.info("Moving for id %s took %f", id, end - start)
            if ret_code:
                _log.error("Failed to move %s from tmp location", id)
                remove_tmp_files()
                return 'move-tmp-failed'
            success = True
        except Exception as e:
            _log.error("Failed to download %s for %s", id, e)

            cause = e.exc_info[1]
            # Too many requests, wait and retry
            if isinstance(cause, HTTPError) and e.exc_info[1].code == 429:
                if retries < max_retries:
                    sleep_hours = retry_hours[retries]
                    sleep_mins = sleep_hours * 60 + 5
                    retries += 1
                    _log.info("Received 429 (too many requests), waiting for {} hour(s), retry {}/{}".format(
                        sleep_hours, retries, max_retries))
                    _log.info("TOTAL DOWNLOAD ATTEMPTS: {}".format(total_download_attempts))
                    # Launch stress
                    stress_proc = launch_stress()
                    time.sleep(sleep_mins * 60)
                    if stress_proc:
                        stop_stress(stress_proc)
                    continue

            time.sleep(10)
            remove_tmp_files()
            return 'download-failed'

    return 'download-succeeded'
