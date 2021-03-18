import os
import numpy as np
from gluoncv.torch import model_zoo
from gluoncv.torch.engine.config import get_cfg_defaults
import torch
import torchvision.transforms as T
from PIL import Image
import tvm
from tvm import relay
from tvm.contrib.download import download
from tvm.runtime.vm import VirtualMachine

# def debug_trace(kpt_bases):
#     stride = 1
#     # a = kpt_bases[:, [1, 0], :, :].contiguous() * stride + stride // 2
#     a = torch.cat((kpt_bases[:, 1:2], kpt_bases[:, 0:1]), dim=1) * stride + stride // 2
#     print(a.shape)
#     return a

from tvm.relay.frontend.pytorch import _op, AttrCvt, get_relay_op
def nms(inputs, input_types):
    boxes = inputs[0]
    scores = inputs[1]
    iou_threshold = inputs[2]

    # TVM NMS assumes score > 0
    scores = scores - _op.min(scores) + _op.const(1.0)

    num_boxes = _op.shape_of(scores)
    # PyTorch NMS doesn't have score_threshold, so no need to run get_valid_count
    indices = _op.transform.arange(_op.squeeze(num_boxes), dtype="int32")
    indices_value = _op.cast(indices, dtype="float32")
    indices_value = _op.expand_dims(indices_value, -1, 1)
    indices = _op.expand_dims(indices, 0, 1)

    # Generate data with shape (1, num_anchors, 5)
    scores = AttrCvt(op_name="expand_dims", extras={"axis": -1, "num_newaxis": 1})([scores], {})
    data = _op.concatenate([scores, boxes, indices_value], -1)
    data = _op.expand_dims(data, 0, 1)

    # Perform Non-Maximum Suppression,
    # PyTorch NMS doesn't have parameter top_k and max_output_size
    score_index = 0
    top_k = max_out_size = -1
    nms_ret = get_relay_op("non_max_suppression")(
        data=data,
        valid_count=num_boxes,
        indices=indices,
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
    # squeeze the two outputs of nms for strided_slice
    size = get_relay_op("squeeze")(nms_ret[1], axis=[1])
    data_slice = get_relay_op("squeeze")(nms_ret[0], axis=[0])

    # strided slice to get the dynamic result
    ret = get_relay_op("strided_slice")(
        data_slice, begin=_expr.const([0]), end=size, slice_mode="size"
    )
    # in torchvision, indices from nms are int64
    return _op.cast(ret, "int64")

def export_onnx(model, input):
    torch.onnx.export(model, input, 'model.onnx')

def get_image(img_name='street_small.jpg', img_url=None):
    def get_single_image_input(img_url, save_dir='/tmp'):
        img_name = img_url.split("/")[-1]
        img_path = os.path.join(save_dir, img_name)
        download(img_url, img_path)
        orig_img = Image.open(img_path)
        # img = orig_img.resize((736, 1280), Image.LANCZOS)
        img = orig_img.resize((1280, 800), Image.LANCZOS)
        img = np.array(img)[:, :, (2, 1, 0)]
        return img, orig_img, img_path

    def get_transforms():
        tforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.00392157, 0.00392157, 0.00392157])])
        return tforms
    
    if img_url is None:
        img_url = f"https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/{img_name}"
    img, orig_img, img_path = get_single_image_input(img_url)

    tforms = get_transforms()
    input_data = tforms(img).unsqueeze(0)
    return input_data, orig_img

def convert_pt_to_tvm_type(idtype):
    """ Accepts a pytorch dtype and returns string TVM dtype."""
    # TVM does not support PyTorch complex dtypes
    if idtype == torch.float64:
        curr_dtype = "float64"
    elif idtype == torch.float32:
        curr_dtype = "float32"
    elif idtype == torch.float16:
        curr_dtype = "float16"
    elif idtype == torch.bfloat16:
        curr_dtype = "bfloat16"
    elif idtype == torch.int64:
        curr_dtype = "int64"
    elif idtype == torch.int32:
        curr_dtype = "int32"
    elif idtype == torch.int16:
        curr_dtype = "int16"
    elif idtype == torch.int8:
        curr_dtype = "int8"
    elif idtype == torch.uint8:
        curr_dtype = "uint8"
    elif idtype == torch.bool:
        curr_dtype = "bool"
    else:
        raise NotImplementedError("Unsupported dtype: {}".format(idtype))
    return curr_dtype

def verify_model_vm(input_model, ishapes, idtype=None, idata=None, targets=["llvm"]):
    if not idtype:
        idtype = torch.float

    input_names = ["i{}".format(idx) for idx, ish in enumerate(ishapes)]
    tvm_dtype = convert_pt_to_tvm_type(idtype)
    input_dtypes = [tvm_dtype] * len(input_names)
    input_shapes = list(zip(input_names, list(zip(ishapes, input_dtypes))))

    if idata is not None:
        input_data = idata
    # If no input_data provided, generate random data of specified dtype
    else:
        if idtype == torch.bool:
            input_data = [
                torch.Tensor.bool(torch.randint(low=0, high=2, size=shape)) for shape in ishapes
            ]
        # Torch dtype can be float, complex, int, or Bool. Complex not supported, so if not float or Bool,
        # dtype must be int!
        elif not idtype.is_floating_point:
            input_data = [
                torch.randint(low=0, high=10, size=shape, dtype=idtype) for shape in ishapes
            ]
        else:
            input_data = [torch.randn(shape, dtype=idtype) for shape in ishapes]

    # Compile via VM
    mod, params = relay.frontend.from_pytorch(input_model, input_shapes)

    for tgt in targets:
        print("Running on target", tgt)
        ctx = tvm.context(tgt, 0)

        executor = relay.create_executor("vm", mod=mod, ctx=ctx, target=tgt)
        evaluator = executor.evaluate()

        # Inference
        for name, inp in zip(input_names, input_data):
            params[name] = inp.numpy()
        vm_res = evaluator(**params)

        # Baseline result
        with torch.no_grad():
            pt_result = input_model(*input_data)

        # Verify the accuracy
        if isinstance(pt_result, tuple):
            # handle multiple outputs
            for i in range(len(pt_result)):
                tvm_res = vm_res[i].asnumpy()
                tvm.testing.assert_allclose(tvm_res, pt_result[i].numpy(), rtol=1e-5, atol=1e-5)
        elif not isinstance(pt_result, torch.Tensor):
            tvm_res = vm_res.asnumpy().item()
            assert pt_result == tvm_res
        else:
            tvm.testing.assert_allclose(vm_res.asnumpy(), pt_result.numpy(), rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    device = torch.device('cuda')
    cfg = get_cfg_defaults()
    # cfg.merge_from_file('./configurations/ms_dla_34_4x_syncbn.yaml')
    # net = model_zoo.dla34_fpn_directpose(cfg).to(device).eval()
    # model = torch.load('model_final.pth')['model']
    cfg.merge_from_file('./configurations/ms_aa_resnet50_4x_syncbn.yaml')
    net = model_zoo.resnet_lpf_fpn_directpose(cfg).to(device).eval()
    model = torch.load('model_final_resnet.pth')['model']
    print(model['pixel_mean'])
    print(model['pixel_std'])
    # np_model = {k: v.cpu().numpy() for k, v in model.items()}
    # import numpy as np
    # np.save('np_model.npy', np_model)
    # raise
    net.load_state_dict(model, strict=False)
    # images = torch.zeros(1, 3, 512, 512).cuda()
    images, orig_image = get_image('soccer.png', img_url='https://github.com/dmlc/web-data/blob/master/gluoncv/pose/soccer.png?raw=true')
    y = net(images.to(device))
    # print(y[0], y[0].shape)
    # for yy in y:
    #     print(yy.shape, yy)
    # raise
    # with torch.no_grad():
    #     scripted_model = torch.jit.trace(debug_trace, torch.zeros(1, 2, 48, 48))
    # torch._C._jit_pass_inline(scripted_model.graph)
    # print(scripted_model.graph)
    # export_onnx(net, images)
    # raise
    with torch.no_grad():
        scripted_model = torch.jit.trace(net.forward, images.to(device)).eval()

    # torch._C._jit_pass_inline(scripted_model.graph)
    # print(scripted_model.graph)
    input_name = "input0"
    shape_list = [(input_name, images.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, {"torchvision::nms": nms})
    # func = mod['main']
    # cid = relay.TupleGetItem(func, 0)
    # scores = relay.TupleGetItem(func, 1)
    # bboxes = relay.TupleGetItem(func, 2)
    # keypoints = relay.TupleGetItem(func, 3)
    # num_boxes = relay.shape_of(scores)
    # indices = relay.arange(relay.squeeze(num_boxes), dtype="int32")
    # indices = relay.expand_dims(indices, 0, 1)
    # scores = relay.expand_dims(scores, -1, 1)
    # data = relay.concatenate([scores, bboxes], -1)
    # data = relay.expand_dims(data, 0, 1)
    
    # nms_ret = relay.vision.non_max_suppression(
    #     data=bboxes, 
    #     valid_count=num_boxes,
    #     indices=indices,
    #     max_output_size=-1,
    #     iou_threshold=cfg.CONFIG.MODEL.DIRECTPOSE.NMS_TH,
    #     force_suppress=True,
    #     top_k=-1,
    #     coord_start=1,
    #     score_index=0,
    #     id_index=-1,
    #     return_indices=False,
    #     invalid_to_bottom=False,
    # )
    # nms_ret = relay.squeeze(nms_ret, axis=[0])
    # new_bboxes = relay.strided_slice(nms_ret, begin=relay.const([0, 1]), end=relay.const([-1, 5]))
    # new_scores = relay.strided_slice(nms_ret, begin=relay.const([0, 0]), end=relay.const([-1, 1]))
    # new_out = relay.Tuple([cid, new_scores, new_bboxes, keypoints])
    # new_func = relay.Function(func.params, new_out, None, func.type_params, func.attrs)
    # mod = tvm.ir.IRModule({"main": new_func})
    # print(new_func)
    print(mod)

    target = "cuda"
    target_host = "llvm"
    ctx = tvm.gpu(0)


    # verify_model_vm(scripted_model, [data.shape for data in [images]], idata=[images.to(device)], idtype=None, targets=target)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)
        # vm = relay.vm.compile(mod, target=target, params=params)
    export_graph, export_lib, export_params = lib
    export_lib.export_library('compiled.so')
    with open('compiled.json', 'w') as f:
        f.write(export_graph)
    with open('compiled.params', 'wb') as f:
        f.write(relay.save_param_dict(export_params))
    print('export complete')
    raise

    from tvm.contrib import graph_runtime

    dtype = "float32"
    # m = VirtualMachine(vm, ctx)
    m = graph_runtime.GraphModule(lib["default"](ctx))
    # Set inputs
    m.set_input(input_name, tvm.nd.array(images.cpu().detach().numpy().astype(dtype)))
    # m.set_input("main", **{input_name: tvm.nd.array(images.cpu().detach().numpy().astype(dtype))})
    # Execute
    tvm_result = m.run()
    # Get outputs
    # for ii in range(3):
    #     tvm_output = m.get_output(ii)
    #     # tvm_output = tvm_result[ii].asnumpy()
    #     print(tvm_output.shape, tvm_output)
    ids = m.get_output(0).asnumpy()
    xxx = m.get_output(1).asnumpy()
    kpts = m.get_output(2).asnumpy()
    scores = xxx[:, 0]
    bboxes = xxx[:, 1:5]
    orig_indices = xxx[:, 5]
    valid = np.where(np.logical_and(scores > 0.5, orig_indices > -0.5))[0]
    print(valid)
    print(bboxes[valid, :])
    print(scores[valid])
    new_idx = orig_indices[valid].astype(int)
    print('new_idx', new_idx)
    print(kpts[new_idx, :])
    print(ids[new_idx])
