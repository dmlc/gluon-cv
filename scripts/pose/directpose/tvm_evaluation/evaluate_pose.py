import json
import numpy as np
import os
from pose_model import PoseEstimationInferenceModel
import cv2
from PIL import Image

from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_INFO = {
    'MSCOCOPerson2017_val': {
        "img_dir": os.path.expanduser('~/.mxnet/datasets/coco/val2017'),
        "ann_file": os.path.expanduser('~/.mxnet/datasets/coco/annotations/person_keypoints_val2017.json')
    },
}

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Pose Evaluation')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--load-result', action='store_true')
    parser.add_argument('--gpus', type=int, default=8)
    parser.add_argument('--val-annotations', type=str, nargs='+', default=['MSCOCOPerson2017_val',], help='MSCOCOPerson2017_val and IVISION_val')
    parser.add_argument('--model-prefix', type=str, default='./compiled')
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=800)
    parser.add_argument("--export-dir", type=str, default='./')
    parser.add_argument("--num-classes", type=int, default=1)
    args = parser.parse_args()
    return args

def _get_thr_ind(coco_eval, thr):
    ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                   (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
    iou_thr = coco_eval.params.iouThrs[ind]
    assert np.isclose(iou_thr, thr)
    return ind

def draw_skeleton(img, kp, thresh=0.2, color=None):
    """Draws a COCO-style skeleton defined by keypoints `kp` on the current axes."""
    SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
    MARGIN = 2
    num_joints = 17

    if color is None:
        color = np.random.randint(0, 256, (1, 3), dtype=np.uint8).tolist()[0]

    x = kp[:, 0]
    y = kp[:, 1]
    scores = kp[:, 2]

    # Remove predictions outside the image
    for i in range(len(scores)):
        if (x[i] <= MARGIN) or (x[i] >= img.shape[1] - MARGIN) or (y[i] <= MARGIN) or (
                y[i] >= img.shape[0] - MARGIN):
            scores[i] = 0

    # Only display predictions with score greater than a threshold
    indxs = scores > thresh

    im_scale = min(img.shape[:2])
    for p in SKELETON:
        if (indxs[p[0]]) and (indxs[p[1]]):
            cv2.line(img, (int(x[p[0]]), int(y[p[0]])), (int(x[p[1]]), int(y[p[1]])), color,
                     int(im_scale / 300) + 1)

    for i in range(num_joints):
        if indxs[i]:
            cv2.circle(img, (int(x[i]), int(y[i])), 2, color, 2)
    return img

def viusalize(image, result, output_path):
    if image["transform_info"] is None:
        image = image["image"]
    else:
        # mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)
        # std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)
        mean = np.array([0.406 * 255, 0.456 * 255, 0.485* 255], dtype=np.float32)
        std = np.array([1, 1, 1], dtype=np.float32)
        p_b_w, p_b_h, padded_h, padded_w = image["transform_info"]
        image = image["image"][0]
        image = np.transpose(image, (1, 2, 0))
        image = cv2.resize(image, (padded_w, padded_h), interpolation=cv2.INTER_LINEAR)
        image = image[p_b_h:padded_h-p_b_h, p_b_w:padded_w-p_b_w, :]
        image = (image * std + mean).astype(np.int)

    # image = image[:, :, ::-1].astype(np.uint8)
    image = image.astype(np.uint8)
    bboxes = result['bboxes']
    scores = result['scores']
    keypoints = result['keypoints']
    for bbox, kpt, score in zip(bboxes, keypoints, scores):
        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        image = draw_skeleton(image, kpt, color=(255, 0, 0))
    cv2.imwrite(output_path, image)
    return

def main():
    args = parse_args()
    load_result = args.load_result
    num_gpus = args.gpus

    if not load_result:
        index = args.index
        model = PoseEstimationInferenceModel(model_prefix=args.model_prefix,
                                              gpu_id=index,
                                              image_width=args.image_width,
                                              image_height=args.image_height,
                                              threshold=0.05)

        for dataset_name in args.val_annotations:
            gt_file = DATASET_INFO[dataset_name]['ann_file']
            img_dir = DATASET_INFO[dataset_name]['img_dir']

            val_imgs = json.load(open(gt_file))['images']
            start_index = (len(val_imgs) // num_gpus + 1) * index
            end_index = (len(val_imgs) // num_gpus + 1) * (index + 1)

            coco_results = []
            count = 0
            for img_name in tqdm(val_imgs[start_index:end_index]):
                # Test on image, change from BGR channel to RGB channel
                raw_image = cv2.imread(os.path.join(img_dir, img_name['file_name']))
                # raw_image = np.array(Image.open(os.path.join(img_dir, img_name["file_name"])))[:, :, ::-1]
                #raw_image = raw_image[:, :, ::-1]
                image = {
                    "image": raw_image,
                    "transform_info": None
                }

                result = model.process([image])
                # print(result)
                # viusalize(image, result[0], output_path=img_name['file_name'])
                # count += 1
                # if count == 5:
                #     raise
                # raise
                for bbox, cat_id, score, kpt in zip(result[0]['bboxes'], result[0]['class_ids'], result[0]['scores'],
                                                    result[0]['keypoints']):
                    coco_results.append({'image_id': img_name['id'],
                                          'category_id': int(cat_id + 1),
                                          'keypoints': kpt.flatten().astype(float).tolist(),
                                          'bbox': bbox.flatten().astype(float).tolist(),
                                          'score': float(score[0])})

            os.makedirs(os.path.join(args.export_dir, dataset_name), exist_ok=True)
            with open(os.path.join(args.export_dir, dataset_name, '{}_end2endtest.json'.format(index)), 'w') as f:
                json.dump(coco_results, f)
    else:
        for dataset_name in args.val_annotations:
            gt_file = DATASET_INFO[dataset_name]['ann_file']

            # Aggregate All end2end test result and filter low confidence person detections
            if not os.path.exists(os.path.join(args.export_dir, dataset_name)):
                print("Please run distribute_test.sh first to save prediction results!")
                return

            res_figure = os.path.join(args.export_dir, dataset_name, 'PRCurve.png')
            res_file = os.path.join(args.export_dir, dataset_name, 'final_res.json')
            if not os.path.exists(res_file):
                final_res = []
                for index in range(num_gpus):
                    final_res.extend(json.load(open(os.path.join(args.export_dir, dataset_name, '{}_end2endtest.json'.format(index)), 'r')))
                with open(res_file, 'w') as f:
                    json.dump(final_res, f)

            # COCO based evaluation
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            coco_gt = COCO(gt_file)
            coco_dt = coco_gt.loadRes(res_file)
            coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Per category evaluation PR Curve
            IoU_lo_thresh = 0.5
            ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)

            num_class = args.num_classes
            precision = coco_eval.eval['precision'][ind_lo, :, :num_class, 0, 0]
            mAP = np.mean(precision[precision > -1])
            print('mAP @ OKS={:.2f} : {:.4f}'.format(IoU_lo_thresh, mAP))

            recall = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
            color = [('b-', 'person')]

            for cls_ind in range(num_class):
                mAP = coco_eval.eval['precision'][:, :, cls_ind, 0, 0]
                mAP = np.mean(mAP[mAP > -1])
                if not np.isnan(mAP):
                    print('Class {} mAP @ OKS=0.5:0.95 : {:.4f}'.format(cls_ind, mAP))
                prec = precision[:, cls_ind]
                ap = np.mean(prec[prec > -1])
                if not np.isnan(ap):
                    print('Class {}: {:.4f}'.format(cls_ind, ap))
                    plt.plot(recall, prec, color[cls_ind][0], label=color[cls_ind][1])

            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.legend()
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.savefig(res_figure)

if __name__ == '__main__':
    main()
