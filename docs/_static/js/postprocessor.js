class Postprocessor {
    constructor(task) {
        this.task = task;
        this.color_maps = {};
    }

    get_random_color() {
        var letters = '0123456789ABCDEF';
        var color = '#';
        for (var i = 0; i < 6; i++) {
          color += letters[Math.floor(Math.random() * 16)];
        }
        return color;
    }

    scores_to_probs(scores) {
        var probs = [];
        const softmax_sum = scores.reduce((a, b) => a + Math.exp(b), 0);
        scores.forEach((score) => probs.push(Math.exp(score) / softmax_sum));
        return probs;
    }

    process_classification(scores, ctx) { 
        var k = ctx.k;
        if (!k) { k = 5; }
        const probs = this.scores_to_probs(Array.from(scores));
        const probs_indices = probs.map(
            function (prob, index) {
            return [prob, index];
            }
        );
        const sorted = probs_indices.sort(
            function (a, b) {
                if (a[0] < b[0]) {
                    return -1;
                }
                if (a[0] > b[0]) {
                    return 1;
                }
                return 0;
                }
        ).reverse();
        const topK = sorted.slice(0, k).map(function (prob_index) {
            const i_class = model.classes[prob_index[1]];
            return {
                label: i_class,
                prob: (prob_index[0] * 100).toFixed(2),
                index: prob_index[1],
            };
        });
        return topK;
    }

    remap_bbox(bbox, video_width, video_height, input_width, input_height) {
        const xmin = bbox[0];
        const ymin = bbox[1];
        const xmax = bbox[2];
        const ymax = bbox[3];
        const new_xmin = Math.max(Math.round(xmin * (video_width / input_width)), 0);
        const new_ymin = Math.max(Math.round(ymin * (video_height / input_height)), 0);
        const new_xmax = Math.min(Math.round(xmax * (video_width / input_width)), video_width);
        const new_ymax = Math.min(Math.round(ymax * (video_height / input_height)), video_height);
        const new_bbox_width = new_xmax - new_xmin;
        const new_bbox_height = new_ymax - new_ymin;

        // js requires starting point, width, height to draw rect
        return [new_xmin, new_ymin, new_bbox_width, new_bbox_height];
    }

    process_object_detection(result, ctx) {
        const class_ids = result[0];
        const scores = result[1];
        const bboxes = result[2];
        if ( class_ids.length != bboxes.length/4) {
            throw `The length of labels and bboxes must match ${class_ids.length} vs ${bboxes.length/4}`;
        }
        if (scores.length != bboxes.length/4) {
            throw `The length of scores and bboxes must match ${scores.length} vs ${bboxes.length/4}`;
        }
        const num_result = class_ids.length;
        var video_width = ctx.video_width;
        var video_height = ctx.video_height;
        var input_width = ctx.input_width;
        var input_height = ctx.input_height;
        var threshold = ctx.threshold;
        if (!video_width) { video_width = 512; }
        if (!video_height) { video_height = 512; }
        if (!threshold) { threshold = 0.5; }

        var processed_class_ids = [];
        var processed_scores = [];
        var processed_bboxes = [];

        for (var i = 0; i < num_result; i++) {
            if (scores[i] < threshold) { continue; }
            if (class_ids[i] < threshold) { continue; }
            const bbox = [
                            Math.round(bboxes[i*4+0]), 
                            Math.round(bboxes[i*4+1]), 
                            Math.round(bboxes[i*4+2]), 
                            Math.round(bboxes[i*4+3])
                        ];
            processed_class_ids.push(model.classes[class_ids[i]]);
            processed_scores.push(scores[i].toFixed(3));
            processed_bboxes.push(this.remap_bbox(bbox, video_width, video_height, input_width, input_height));
            if (!(model.classes[class_ids[i]] in this.color_maps)) { 
                this.color_maps[model.classes[class_ids[i]]] = this.get_random_color(); 
            }
        }

        return [processed_class_ids, processed_scores, processed_bboxes, this.color_maps];

    }

    process(result, ctx) {
        switch (this.task) {
            case tasks.CLASSIFICATION:
                return this.process_classification(result, ctx);
            case tasks.OBJECT_DETECTION:
                return this.process_object_detection(result, ctx);
            case tasks.SEMANTIC_SEGMENTATION:
                break;
            case tasksk.INSTANCE_SEGMENTATION:
                break;
            case ttasks.POSE_ESTIMATION:
                break;
        }
    }
}
