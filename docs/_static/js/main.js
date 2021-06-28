const tasks = {
    CLASSIFICATION: 'classification',
    OBJECT_DETECTION: 'object_detection',
    SEMANTIC_SEGMENTATION: 'semantic_segmentation',
    INSTANCE_SEGMENTATION: 'instance_segmentation',
    POSE_ESTIMATION: 'pose_estimation',
}

var media_constraints = {
    video: {
        width: 720,
        height: 480,
    },
    audio: false,
};
var current_stream;

var model_path;
var task;
var model;
var classification_session;
var detection_session;
var postprocessor;
const preprocessor = new Preprocessor();

function block_ui_on_loading() {
    const predict_button = document.getElementById('predict_button');
    const classification_tab = document.getElementById('classification_tab');
    const object_detection_tab = document.getElementById('object_detection_tab');
    predict_button.innerHTML = 'Loading';
    predict_button.disabled = true;
    classification_tab.disabled = true;
    object_detection_tab.disabled = true;
}

function unblock_ui_on_loading() {
    const predict_button = document.getElementById('predict_button');
    const classification_tab = document.getElementById('classification_tab');
    const object_detection_tab = document.getElementById('object_detection_tab');
    predict_button.innerHTML = 'Predict';
    predict_button.disabled = false;
    classification_tab.disabled = false;
    object_detection_tab.disabled = false;
}

async function on_classification() {
    if (task == tasks.CLASSIFICATION) { return; }
    processor.clear();
    block_ui_on_loading();
    model_path = 'https://apache-mxnet.s3-us-west-2.amazonaws.com/onnx/models/gluoncv-mobilenetv3_large-ad683fdc.onnx';
    // model_path = 'https://damp-mountain-14992.herokuapp.com/https://apache-mxnet.s3-us-west-2.amazonaws.com/onnx/models/gluoncv-mobilenetv3_large-ad683fdc.onnx';
    // model_path = '_static/models/mobilenetv3_large.onnx' //local test only
    task = tasks.CLASSIFICATION;
    model = new Model(model_path, 224, 224, task, image_net_labels);
    postprocessor = new Postprocessor(model.task);
    // the website won't cache the model for some reason, resulting redownload each time
    if (classification_session === undefined) {
        await ort.InferenceSession.create(model.path).then((session) => {
            classification_session = session;
            unblock_ui_on_loading();
        });
    } else {
        unblock_ui_on_loading();
    }
}

async function on_obj_detection() {
    if (task == tasks.OBJECT_DETECTION) { return; }
    processor.clear();
    block_ui_on_loading();
    model_path = 'https://apache-mxnet.s3-us-west-2.amazonaws.com/onnx/models/gluoncv-yolo3_mobilenet1.0_coco-115299e3.onnx';
    // model_path = 'https://damp-mountain-14992.herokuapp.com/https://apache-mxnet.s3-us-west-2.amazonaws.com/onnx/models/gluoncv-yolo3_mobilenet1.0_coco-115299e3.onnx';
    // model_path = '_static/models/yolo3_mobilenet1.0_voc.onnx' //local test only
    task = tasks.OBJECT_DETECTION;
    model = new Model(model_path, 512, 512, task, coco_detection_labels);
    postprocessor = new Postprocessor(model.task);
    // the website won't cache the model for some reason, resulting redownload each time
    if (detection_session === undefined) {
        await ort.InferenceSession.create(model.path).then((session) => {
            detection_session = session;
            unblock_ui_on_loading();
        });
    } else {
        unblock_ui_on_loading();
    }
}

function predict() {
    processor.computeFrame();
}

var processor = {
    did_load: false,

    do_load: function() {
        this.video = document.getElementById('local_video_stream');
        this.video_width = this.video.width;
        this.video_height = this.video.height;
        this.canvas = document.getElementById('result_canvas');
        this.canvas_ctx = this.canvas.getContext('2d', { alpha: false });
        this.did_load = true;
    },

    show_classification_result: function(label, prob) {
        var classification_left_element = document.getElementById('classification_left');
        var classification_right_element = document.getElementById('classification_right');

        var label_element = document.createElement('p');
        label_element.innerHTML = label;
        label_element.className = 'classification_label';
        classification_left_element.appendChild(label_element);

        var div = document.createElement('div');
        var prob_bar_element = document.createElement('progress');
        prob_bar_element.className = 'classification_prob_bar';
        prob_bar_element.value = prob;
        prob_bar_element.max = '100';
        var prob_element = document.createElement('p');
        prob_element.innerHTML = prob + '%';
        prob_element.className = 'classification_prob';

        div.appendChild(prob_bar_element);
        div.appendChild(prob_element);
        classification_right_element.appendChild(div);
    },

    draw_bbox(label, score, bbox, color) {
        [xmin, ymin, width, height] = bbox;
        this.canvas_ctx.strokeStyle = color;
        this.canvas_ctx.fillStyle = color;
        this.canvas_ctx.lineWidth = 3;
        this.canvas_ctx.font = "30px Comic Sans MS";
        this.canvas_ctx.strokeRect(xmin, ymin, width, height);
        if (ymin-30 <= 0) {
            this.canvas_ctx.fillText(`${label} ${score}`, xmin, ymin+30);
        } else {
            this.canvas_ctx.fillText(`${label} ${score}`, xmin, ymin-10);
        }
    },

    visualize: function(processed_result) {
        var classification_results_element = document.getElementById('classification_results');
        classification_results_element.style.visibility = 'hidden';
        switch (model.task) {
            case tasks.CLASSIFICATION:
                var classification_results_element = document.getElementById('classification_results');
                var classification_left_element = document.getElementById('classification_left');
                var classification_right_element = document.getElementById('classification_right');
                classification_left_element.innerHTML = '';
                classification_right_element.innerHTML = '';
                classification_results_element.style.visibility = 'visible';
                processed_result.map((r) => this.show_classification_result(r.label, r.prob));
                break;
            case tasks.OBJECT_DETECTION:
                [classes, scores, bboxes, color_maps] = processed_result;
                for (var i = 0; i < classes.length; i++) {
                    this.draw_bbox(classes[i], scores[i], bboxes[i], color_maps[classes[i]]);
                }
                break;
            default:
                alert('Error: task ' + model.task + ' has not been implemented');
                break;
        }
    },

    computeFrame: async function() {
        if (this.video.paused || this.video.ended) {
            return;
        }
        this.canvas_ctx.drawImage(this.video, 0, 0, model.input_width, model.input_height);
        var frame = this.canvas_ctx.getImageData(0, 0, model.input_width, model.input_height);
        this.canvas_ctx.drawImage(this.video, 0, 0, this.video_width, this.video_height);
        var frame_length = frame.data.length / 4;
        var rgba_frame_f32 = Float32Array.from(frame.data);
        var rgb_frame_f32 = preprocessor.remove_alpha_channel(rgba_frame_f32, frame_length);

        const image_tensor = new ort.Tensor('float32', rgb_frame_f32, [1,model.input_width,model.input_height,3]);
        var result;
        var data;
        // extract the data from result and visualize
        switch (model.task) {
            case tasks.CLASSIFICATION:
                result = await classification_session.run({data: image_tensor});
                data = Object.keys(result).map((key) => result[key])[0].data;
                this.visualize(postprocessor.process(data, { k:5 }));
                break;
            case tasks.OBJECT_DETECTION:
                result = await detection_session.run({data: image_tensor});
                data = Object.keys(result).map((key) => result[key].data);
                this.visualize(postprocessor.process(data, { 
                                                video_width: this.video_width, 
                                                video_height: this.video_height,
                                                input_width: model.input_width,
                                                input_height: model.input_height,
                                                threshold: 0.5,
                                                }
                                            ));

                break;
            default:
                alert('Error: task ' + model.task + ' has not been implemented');
                break;
        }

        return;
    },

    clear: function() {
        if (this.canvas_ctx) {
            this.canvas_ctx.clearRect(0, 0, this.video_width, this.video_height);
            this.canvas_ctx.rect(0, 0, this.video_width, this.video_height);
            this.canvas_ctx.fillStyle = 'black';
            this.canvas_ctx.fill();
        }
        var classification_results_element = document.getElementById('classification_results');
        classification_results_element.style.visibility = 'hidden';
    }
};

function got_devices(mediaDevices) {
    const camera_select = document.getElementById('camera_select');
    camera_select.innerHTML = '';
    let count = 1;
    mediaDevices.forEach(mediaDevice => {
        if (mediaDevice.kind === 'videoinput') {
            const option = document.createElement('option');
            option.value = mediaDevice.deviceId;
            const label = mediaDevice.label || `Camera ${count++}`;
            const textNode = document.createTextNode(label);
            option.appendChild(textNode);
            camera_select.appendChild(option);
        }
    });
}

function prepare_devices() {
    navigator.mediaDevices.enumerateDevices()
    .then(got_devices)
    .then(get_camera);
}

function get_camera() {
    if (typeof current_stream !== undefined) {
        stop_current_stream();
    } 
    const camera_select = document.getElementById('camera_select');
    if (camera_select.value === '') {
        media_constraints.video.facingMode = 'environment';
    } else {
        media_constraints.video.deviceId = { exact: camera_select.value };
    }
    navigator.mediaDevices.getUserMedia(media_constraints)
    .then(function (stream) {
        var local_video_stream = document.getElementById('local_video_stream');
        local_video_stream.srcObject = stream;
        processor.do_load();
        current_stream = stream;
    })
    .then(()=> {
        document.getElementById('video_container').style.visibility = 'visible';
    })
    .catch(handle_get_user_media_error);
}

function stop_current_stream() {
    if (current_stream) {
        current_stream.getTracks().forEach(track => {
            track.stop();
        });
    }
}

function handle_get_user_media_error(e) {
    switch(e.name) {
        case 'NotFoundError':
            alert('Unable to push video because no camera was found');
            break;
        case 'SecurityError':
        case 'PermissionDeniedError':
            break;
        default:
            alert('Error opening your camera: ' + e.message);
            break;
    }
}

async function main() {
    try {
        prepare_devices();
    } catch (e) {
        alert(e);
    }
    
}

$(document).ready(function() {
    $('.tab button').on('click', function() {
        $('.tab button').removeClass('selected');
        $(this).addClass('selected');
    });
    document.getElementById('classification_tab').click(); // provide a default tab
    main();
});
