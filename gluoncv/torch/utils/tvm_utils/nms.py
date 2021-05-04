"""A custom nms module for tvm with static shape"""
# pylint: disable=unused-argument
from tvm import relay
from tvm.relay.frontend.pytorch import _op, AttrCvt, get_relay_op

def nms(inputs, input_types):
    """A static NMS hack for torchvision.nms, it requires the nms to be the last layer of the network"""
    boxes = inputs[0]
    scores = inputs[1]
    iou_threshold = inputs[2]
    score_index = 0
    coord_start = 1
    # PyTorch NMS doesn't have parameter top_k and max_output_size
    top_k = max_out_size = -1

    # TVM NMS assumes score > 0
    scores = scores - _op.min(scores) + _op.const(1.0)

    # a tricky to preserve the original index of bounding boxes, as the nms op will re-arrange the array
    indices = _op.transform.arange(_op.squeeze(_op.shape_of(scores)), dtype="int32")
    indices = _op.cast(indices, dtype="float32")
    indices = _op.expand_dims(indices, -1, 1)

    # More efficient way to run NMS for torchvision.nms
    scores = AttrCvt(op_name="expand_dims", extras={"axis": -1, "num_newaxis": 1})([scores], {})

    data = _op.concatenate([scores, boxes, indices], -1)
    data = _op.expand_dims(data, 0, 1)
    valid_ret = _op.vision.get_valid_counts(
        data, score_threshold=1.05, id_index=5, score_index=score_index
    )

    nms_ret = get_relay_op("non_max_suppression")(
        data=valid_ret[1],
        valid_count=valid_ret[0],
        indices=valid_ret[2],
        max_output_size=max_out_size,
        iou_threshold=iou_threshold,
        force_suppress=True,
        top_k=top_k,
        coord_start=1,
        score_index=score_index,
        id_index=5,
        return_indices=False,
        invalid_to_bottom=False,
    )

    nms_ret = relay.squeeze(nms_ret, axis=[0])
    return nms_ret
