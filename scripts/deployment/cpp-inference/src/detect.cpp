#include "argparse.hpp"
#include "common.hpp"

namespace synset {
static std::vector<std::string> VOC_CLASS_NAMES = {
     "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair",
     "cow", "diningtable", "dog", "horse",
     "motorbike", "person", "pottedplant",
     "sheep", "sofa", "train", "tvmonitor"};

static std::vector<std::string> COCO_CLASS_NAMES = {};
}


int main(int argc, char** argv) {
    cv::Mat image = cv::imread("dog.jpg", 1);
    auto data = AsData(image);
    Symbol net;
    std::map<std::string, NDArray> args, auxs;
    LoadCheckpoint("yolo3_darknet53_voc", 0, &net, &args, &auxs);
    auto ctx = Context::cpu();

    args["data"] = data;
    Executor *exec = net.SimpleBind(
      ctx, args, std::map<std::string, NDArray>(),
      std::map<std::string, OpReqType>(), auxs);


    exec->Forward(false);
    NDArray::WaitAll();
    auto ids = exec->outputs[0].Copy(Context(kCPU, 0));
    auto scores = exec->outputs[1].Copy(Context(kCPU, 0));
    auto bboxes = exec->outputs[2].Copy(Context(kCPU, 0));
    // NDArray::WaitAll();
    auto plt = viz::PlotBbox(image, bboxes, scores, ids, 0.5, synset::VOC_CLASS_NAMES, std::map<int, cv::Scalar>(), true);
    cv::imshow("plot", plt);
    cv::waitKey();
    delete exec;
    MXNotifyShutdown();
    return 0;
}