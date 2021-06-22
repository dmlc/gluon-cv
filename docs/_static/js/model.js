class Model {
    constructor(model_path, model_input_width, model_input_height, model_task, classes) {
        this.path = model_path;
        this.input_width = model_input_width;
        this.input_height = model_input_height;
        this.task = model_task;
        this.classes = classes;
    }
}
