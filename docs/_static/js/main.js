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
var session;
var postprocessor;
const preprocessor = new Preprocessor();

async function on_classification() {
    if (task == tasks.CLASSIFICATION) { return; }
    // if (processor.did_load) { processor.clear(); }
    model_path = '_static/models/resnet18_v1.onnx';
    task = tasks.CLASSIFICATION;
    model = new Model(model_path, 224, 224, task, image_net_labels);
    postprocessor = new Postprocessor(model.task);
    session = await ort.InferenceSession.create(model.path);
}

async function on_obj_detection() {
    if (task == tasks.OBJECT_DETECTION) { return; }
    // if (processor.did_load) { processor.clear(); }
    model_path = '_static/models/yolo3_mobilenet1.0_voc.onnx';
    task = tasks.OBJECT_DETECTION;
    model = new Model(model_path, 512, 512, task, voc_detection_labels);
    postprocessor = new Postprocessor(model.task);
    session = await ort.InferenceSession.create(model.path);
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
        var classification_result_element = document.getElementById('classification_result');
        classification_result_element.hidden = true;
        switch (model.task) {
            case tasks.CLASSIFICATION:
                var classification_result_element = document.getElementById('classification_result');
                classification_result_element.hidden = false;
                results = [];
                results.push(`Top ${processed_result.length} classification results are`);
                processed_result.map((r) => results.push(r.name));
                classification_result_element.innerHTML = results.join('<br>');
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
        const result = await session.run({data: image_tensor});
        var data = undefined;
        // extract the data from result and visualize
        switch (model.task) {
            case tasks.CLASSIFICATION:
                data = Object.keys(result).map((key) => result[key])[0].data;
                this.visualize(postprocessor.process(data, { k:5 }));
                break;
            case tasks.OBJECT_DETECTION:
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
        this.canvas_ctx.clearRect(0, 0, this.video_width, this.video_height);
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
    if (typeof current_stream !== 'undefined') {
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
