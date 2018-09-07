#include <opencv2/opencv.hpp>
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

int main(int argc, char** argv) {
    LOG(INFO) << "begin";
    Symbol yolo3 = Symbol::Load("yolo3_darknet53_voc-symbol.json");
    auto data = Symbol::Variable("data");
    LG << "Net loaded";
    std::map<std::string, NDArray> parameters;
    NDArray::Load("yolo3_darknet53_voc-0000.params", 0, &parameters);
    Context ctx = Context::cpu();
    parameters["data"] = NDArray(Shape(1, 416, 416, 3), ctx);
    NDArray::WaitAll();
    LG << "params loaded";

    Executor *exec = yolo3.SimpleBind(
      ctx, parameters, std::map<std::string, NDArray>(),
      std::map<std::string, OpReqType>(), std::map<std::string, NDArray>());

    LG << "exec binded.";

    exec->Forward(false);
    NDArray::WaitAll();
    LG << "infer finished.";
    delete exec;

    MXNotifyShutdown();
    return 0;
}