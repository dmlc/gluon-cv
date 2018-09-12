#include "clipp.hpp"
#include "common.hpp"
#include <chrono>

namespace synset {
static std::vector<std::string> VOC_CLASS_NAMES = {
     "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair",
     "cow", "diningtable", "dog", "horse",
     "motorbike", "person", "pottedplant",
     "sheep", "sofa", "train", "tvmonitor"
 };

static std::vector<std::string> COCO_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static std::vector<std::string> CLASS_NAMES = {};
}

namespace args {
    static std::string model;
    static std::string image;
    static std::string output;
    static std::string class_name_file;
    static int epoch = 0;
    static int gpu = -1;
    static bool quite = false;
    static bool no_display = false;
    static float viz_thresh = 0.3;
}

void ParseArgs(int argc, char** argv) {
    using namespace clipp;

    auto cli = (
        value("model file", args::model),
        value("image file", args::image),
        (option("-o", "--output") & value(match::prefix_not("-"), "outfile", args::output)) % "output image, by default no output",
        (option("--class-file") & value(match::prefix_not("-"), "classfile", args::class_name_file)) % "plain text file for class names, one name per line",
        (option("-e", "--epoch") & integer("epoch", args::epoch)) % "Epoch number to load parameters, by default is 0",
        (option("--gpu") & integer("gpu", args::gpu)) % "Which gpu to use, by default is -1, means cpu only.",
        option("-q", "--quite").set(args::quite).doc("Quite mode, no screen output"),
        option("--no-disp").set(args::no_display).doc("Do not display image"),
        (option("-t", "--thresh") & number("thresh", args::viz_thresh)) % "Visualize threshold, from 0 to 1, default 0.3."
    );
    if (!parse(argc, argv, cli) || args::model.empty() || args::image.empty()) {
        std::cout << make_man_page(cli, argv[0]);
        exit(-1);
    }

    // parse class names
    if (args::class_name_file.empty()) {
        if (EndsWith(args::model, "voc")) {
            if (!args::quite) {
                LOG(INFO) << "Using Pascal VOC names...";
            }
            synset::CLASS_NAMES = synset::VOC_CLASS_NAMES;
        } else if (EndsWith(args::model, "coco")) {
            if (!args::quite) {
                LOG(INFO) << "Using COCO names...";
            }
            synset::CLASS_NAMES = synset::COCO_CLASS_NAMES;
        } else {
            LOG(ERROR) << "Cannot determine class names, ignoring...";
        }
    } else {
        synset::CLASS_NAMES = LoadClassNames(args::class_name_file);
    }
}


int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    // load symbol and parameters
    Symbol net;
    std::map<std::string, NDArray> args, auxs;
    LoadCheckpoint(args::model, args::epoch, &net, &args, &auxs);
    auto ctx = Context::cpu();
    if (args::gpu >= 0) {
        ctx = Context::gpu(args::gpu);
    }

    // load image as data
    cv::Mat image = cv::imread(args::image, 1);
    // resize to avoid huge image, keep aspect ratio
    image = ResizeShortWithin(image, 512, 640, 32);
    if (!args::quite) {
        LOG(INFO) << "Image shape: " << image.cols << " x " << image.rows;
    }

    // set input and bind executor
    auto data = AsData(image);
    args["data"] = data;
    Executor *exec = net.SimpleBind(
      ctx, args, std::map<std::string, NDArray>(),
      std::map<std::string, OpReqType>(), auxs);

    // begin forward
    NDArray::WaitAll();
    auto start = std::chrono::steady_clock::now();
    exec->Forward(false);
    auto ids = exec->outputs[0].Copy(Context(kCPU, 0));
    auto scores = exec->outputs[1].Copy(Context(kCPU, 0));
    auto bboxes = exec->outputs[2].Copy(Context(kCPU, 0));
    NDArray::WaitAll();
    auto end = std::chrono::steady_clock::now();
    if (!args::quite) {
        LOG(INFO) << "Elapsed time {Forward->Result}: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms";
    }

    // draw boxes
    auto plt = viz::PlotBbox(image, bboxes, scores, ids, args::viz_thresh, synset::CLASS_NAMES, std::map<int, cv::Scalar>(), !args::quite);
    
    // display drawn image
    if (!args::no_display) {
        cv::imshow("plot", plt);
        cv::waitKey();
    }

    // output image
    if (!args::output.empty()) {
        cv::imwrite(args::output, plt);
    }
    
    delete exec;
    MXNotifyShutdown();
    return 0;
}