class Preprocessor {
    constructor() {
        this.norm_val = {
            r_mean: 0.485,
            g_mean: 0.456,
            b_mean: 0.406,
            r_std: 0.229,
            g_std: 0.224,
            b_std: 0.225
        };
    }

    remove_alpha_channel(rgba_f32, length) {
        var rgb_frame_f32 = new Float32Array(length * 3);
        for (var i = 0; i < length; i++) {
            rgb_frame_f32[i * 3 + 0] = rgba_f32[i * 4 + 0];
            rgb_frame_f32[i * 3 + 1] = rgba_f32[i * 4 + 1];
            rgb_frame_f32[i * 3 + 2] = rgba_f32[i * 4 + 2];
        }
        return rgb_frame_f32;
    }

    normalize(rgb_frame_f32, l) {
        for (var i = 0; i < l; i++) {
            var r = rgb_frame_f32[i * 3 + 0];
            var g = rgb_frame_f32[i * 3 + 1];
            var b = rgb_frame_f32[i * 3 + 2];
            rgb_frame_f32[i * 3 + 0] = ((r / 255) - this.norm_val.r_mean) / this.norm_val.r_std;
            rgb_frame_f32[i * 3 + 1] = ((g / 255) - this.norm_val.g_mean) / this.norm_val.g_std;
            rgb_frame_f32[i * 3 + 2] = ((b / 255) - this.norm_val.b_mean) / this.norm_val.b_std;
        }
    }
}
