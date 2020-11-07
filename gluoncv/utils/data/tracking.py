"""this script is used to help Process dataset for tracking
mainly crop data like_SiamFC"""
import numpy as np
import sys
from gluoncv.utils.filesystem import try_import_cv2


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    """
    crop image

    Parameters
    ----------
    image: np.array, image
    bbox: np or list, bbox coordinate [xmin,ymin,xmax,ymax]
    out_sz: int , crop image size

    Return:
        crop result
    """
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    cv2 = try_import_cv2()
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

def pos_s_2_bbox(pos, s):
    """
    from center_x,center_y,s to get bbox

    Parameters
    ----------
    pos , x, bbox
    s , int, bbox size

    Return:
        [x_min,y_min,x_max,y_max]
    """
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]

def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instance_size=255, padding=(0, 0, 0)):
    """
    Dataset curation and avoid image resizing during training
    if the tight bounding box has size (w, h) and the context margin is p,
    then the scale factor s is chosen such that the area of the scaled rectangle is equal to a constant
    s(w+2p)×s(h+2p)=A.

    Parameters
    ----------
    image: np.array, image
    bbox: list or np.array, bbox
    context_amount: float, the amount of context to be half of the mean dimension
    exemplar_size: int, exemplar_size
    instance_size: int, instance_size

    Return:
        crop result exemplar z and instance x
    """
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instance_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instance_size, padding)
    return z, x

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
        iteration: int : current iteration.
        total: int, total iterations.
        prefix: str, prefix string.
        suffix: str, suffix string.
        decimals: int, positive number of decimals in percent complete.
        barLength: int, character length of bar.
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()