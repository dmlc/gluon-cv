class Model():
    
    def __init__(self, model_name, input_shape, model_type):
        self.model_name = model_name
        self.model_type = model_type
        if len(input_shape) == 4:
            input_shape.append(input_shape.pop(1)) # change BCHW to BHWC
        self.input_shape = tuple(input_shape)
        self.ctx = mx.cpu(0)
        
        self.param_path = 'params'
        self.exported_model_path = 'exported_models'
        self.exported_model_prefix = 'gluoncv_exported_'
        self.exported_model_suffix = '.onnx'
        self.symbol_suffix = '-symbol.json'
        self.params_suffix = '-0000.params'
        self.onnx_model = os.path.join(self.exported_model_path, 
                                      self.exported_model_prefix+self.model_name+self.exported_model_suffix)
        
    def exists(self):
        return os.path.isfile(self.onnx_model)
    
    def is_3D(self):
        return len(self.input_shape) == 5
        
    def get_model(self):
        self.gluon_model = gcv.model_zoo.get_model(self.model_name, pretrained=True, ctx=self.ctx)
    
    def export(self):
        sym = os.path.join(self.param_path, self.model_name + self.symbol_suffix)
        params = os.path.join(self.param_path, self.model_name + self.params_suffix)
        if self.is_3D():
            export_block(os.path.join(self.param_path,self.model_name), self.gluon_model, 
                         data_shape=self.input_shape[1:], preprocess=False, layout='CTHW')
        else:
            export_block(os.path.join(self.param_path,self.model_name), self.gluon_model, 
                         data_shape=self.input_shape[1:], preprocess=True, layout='HWC')
        # Load model with preprocess layer
        self.gluon_model = SymbolBlock.imports(symbol_file=sym, input_names=['data'],
                                                   param_file=params, ctx=mx.cpu())
            
    def export_onnx(self):
        sym = os.path.join(self.param_path, self.model_name + self.symbol_suffix)
        params = os.path.join(self.param_path, self.model_name + self.params_suffix)
        return onnx_mxnet.export_model(sym, params, [self.input_shape], np.float32, self.onnx_model)
    
    def predict(self, data):
        return self.gluon_model(data)

def test_classification(model):
    test_img_file = os.path.join('test_imgs', 'bikers.jpg')
    img = mx.image.imread(test_img_file)
    img = mx.image.imresize(img, model.input_shape[1], model.input_shape[2])
    img = img.expand_dims(0).astype('float32')

    onnx_session = rt.InferenceSession(model.onnx_model, None)
    input_name = onnx_session.get_inputs()[0].name

    gluon_result = model.predict(img).asnumpy()
    onnx_result = onnx_session.run([], {input_name: img.asnumpy()})[0]
    assert_almost_equal(gluon_result, onnx_result, decimal=3)
    
def assert_obj_detetion_result(gluon_ids, gluon_scores, gluon_boxes,
                                   onnx_ids, onnx_scores, onnx_boxes,
                                   score_thresh=0.6, score_tol=1e-4):
    def assert_bbox(gluon_box, onnx_box, box_tol=1e-2):
        def assert_scalar(a, b, tol=box_tol):
            return np.abs(a-b) <= tol
        return assert_scalar(gluon_box[0], onnx_box[0]) and assert_scalar(gluon_box[1], onnx_box[1]) \
                  and assert_scalar(gluon_box[2], onnx_box[2]) and assert_scalar(gluon_box[3], onnx_box[3])

    found_match = False
    for i in range(len(onnx_ids)):
        onnx_id = onnx_ids[i][0]
        onnx_score = onnx_scores[i][0]
        onnx_box = onnx_boxes[i]

        if onnx_score < score_thresh:
            break
        for j in range(len(gluon_ids)):
            gluon_id = gluon_ids[j].asnumpy()[0]
            gluon_score = gluon_scores[j].asnumpy()[0]
            gluon_box = gluon_boxes[j].asnumpy()
            # check socre 
            if onnx_score < gluon_score - score_tol:
                continue
            if onnx_score > gluon_score + score_tol:
                return False
            # check id
            if onnx_id != gluon_id:
                continue
            # check bounding box
            if assert_bbox(gluon_box, onnx_box):
                found_match = True
                break
        if not found_match:
            return False
        found_match = False
    return True
    
def test_detection(model):
    test_img_file = os.path.join('test_imgs', 'runners.jpg')
    img = mx.image.imread(test_img_file)
    img, _ = mx.image.center_crop(img, size=(model.input_shape[1], model.input_shape[2]))
    img = img.expand_dims(0).astype('float32')

    onnx_session = rt.InferenceSession(model.onnx_model, None)
    input_name = onnx_session.get_inputs()[0].name

    gluon_class_ids, gluon_scores, gluon_boxes = model.predict(img)
    # center_net_resnet models have different output format
    if 'center_net_resnet' in model.model_name:
        onnx_scores, onnx_class_ids, onnx_boxes = onnx_session.run([], {input_name: img.asnumpy()})
        assert_almost_equal(gluon_class_ids.asnumpy(), onnx_class_ids, decimal=3)
        assert_almost_equal(gluon_scores.asnumpy(), onnx_scores, decimal=3)
        assert_almost_equal(gluon_boxes.asnumpy(), onnx_boxes, decimal=3)
    else:
        onnx_class_ids, onnx_scores, onnx_boxes = onnx_session.run([], {input_name: img.asnumpy()})
        if not assert_obj_detetion_result(gluon_class_ids[0], gluon_scores[0], gluon_boxes[0],
                onnx_class_ids[0], onnx_scores[0], onnx_boxes[0]):
            raise AssertionError("Assertion error on model: " + model.model_name)
            
def test_segmentation(model):
    test_img_file = os.path.join('test_imgs', 'runners.jpg')
    img = mx.image.imread(test_img_file)
    img, _ = mx.image.center_crop(img, size=(model.input_shape[1], model.input_shape[2]))
    img = img.expand_dims(0).astype('float32')
    
    onnx_session = rt.InferenceSession(model.onnx_model, None)
    input_name = onnx_session.get_inputs()[0].name
    
    gluon_result = model.predict(img)
    onnx_result = onnx_session.run([], {input_name: img.asnumpy()})

    assert(len(gluon_result) == len(onnx_result))
    for i in range(len(gluon_result)):
        # deeplab_v3b_plus_wideresnet_citys has a different output format
        if model.model_name == 'deeplab_v3b_plus_wideresnet_citys':
            assert_almost_equal(gluon_result[i].asnumpy(), onnx_result[i][0], decimal=3)
        else:
            assert_almost_equal(gluon_result[i].asnumpy(), onnx_result[i], decimal=3)

def test_pose_estimation(model):
    test_img_file = os.path.join('test_imgs', 'runners.jpg')
    img = mx.image.imread(test_img_file)
    img, _ = mx.image.center_crop(img, size=(model.input_shape[1], model.input_shape[2]))
    img = img.expand_dims(0).astype('float32')
    
    onnx_session = rt.InferenceSession(model.onnx_model, None)
    input_name = onnx_session.get_inputs()[0].name
    
    gluon_result = model.predict(img)
    onnx_result = onnx_session.run([], {input_name: img.asnumpy()})[0]
    assert(len(gluon_result) == len(onnx_result))
    for i in range(len(gluon_result)):
        assert_almost_equal(gluon_result[i].asnumpy(), onnx_result[i], decimal=3)

def test_action_recognition(model):
    onnx_session = rt.InferenceSession(model.onnx_model, None)
    input_name = onnx_session.get_inputs()[0].name
    
    if not model.is_3D():
        test_img_file = os.path.join('test_imgs', 'ThrowDiscus.png')
        img = mx.image.imread(test_img_file)
        img, _ = mx.image.center_crop(img, size=(model.input_shape[1], model.input_shape[2]))
        img = img.expand_dims(0).astype('float32')
        
        gluon_result = model.predict(img)
        onnx_result = onnx_session.run([], {input_name: img.asnumpy()})[0]
        assert_almost_equal(gluon_result.asnumpy(), onnx_result, decimal=3)
    else:
        from gluoncv.utils.filesystem import try_import_decord
        decord = try_import_decord()
        video_fname = os.path.join('test_videos', 'abseiling_k400.mp4')
        vr = decord.VideoReader(video_fname)
        frame_id_list = None
        if 'slowfast' in model.model_name:
            fast_frame_id_list = range(0, 64, 2)
            slow_frame_id_list = range(0, 64, 16) if '4x16' in model.model_name else range(0, 64, 8)
            frame_id_list = list(fast_frame_id_list) + list(slow_frame_id_list)
        else:
            frame_id_list = list(range(0, 64, 2))
        num_frames = len(frame_id_list)
        video_data = vr.get_batch(frame_id_list).asnumpy()
        clip_input = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        transform_fn = video.VideoGroupValTransform(size=model.input_shape[3], 
                                                    mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])
        clip_input = transform_fn(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (num_frames, 3, model.input_shape[3], model.input_shape[4]))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        
        gluon_result = model.predict(nd.array(clip_input)).asnumpy()
        onnx_result = onnx_session.run([], {input_name: clip_input.astype('float32')})[0]
        assert_almost_equal(gluon_result, onnx_result, decimal=3)

def test_depth_prediction(model):
    raise Exception("Depth preidction test not implemented yet")
    

