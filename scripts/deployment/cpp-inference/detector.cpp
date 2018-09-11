#include <vector>
#include <opencv2/opencv.hpp>
#include "mxnet-cpp/MxNetCpp.h"
#include "argparse.hpp"

using namespace mxnet::cpp;

void plot_bbox(cv::Mat img, NDArray bboxes, NDArray scores, NDArray labels, float thresh) {
    int num = bboxes.GetShape()[1];
    for (int i = 0; i < num; ++i) {
        float score = scores.At(0, 0, i);
        float label = labels.At(0, 0, i);
        if (score < thresh || label < 0) {
            LG << "score: " << score << " label " << label << ", i: " << i;
            break;
        }

        LG << bboxes.At(0, i, 0) << "," << bboxes.At(0, i, 1) << "," << bboxes.At(0, i, 2) << "," << bboxes.At(0, i, 3);
        cv::Point pt1(bboxes.At(0, i, 0), bboxes.At(0, i, 1));
        // float xmin = ;
        // float ymin = ;
        cv::Point pt2(bboxes.At(0, i, 2), bboxes.At(0, i, 3));
        // float xmax = ;
        // float ymax = bboxes.At(i, 3);
        cv::rectangle(img, pt1, pt2, cv::Scalar(255, 0, 0));
    }
    cv::imshow("debug", img);
    cv::waitKey();
}

int main(int argc, char** argv) {
    LG << "Begin.";
    cv::Mat image = cv::imread("dog.jpg", 1);
    LG << image.rows << " , " << image.cols;
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32FC3);
    cv::Mat flat_image = rgb_image.reshape(1, 1);
    for (int i = 0; i < 10; i++) {
            LG << i << ": " << rgb_image.at<cv::Vec3f>(0, i)[0] << "," << rgb_image.at<cv::Vec3f>(0, i)[1] << "," << rgb_image.at<cv::Vec3f>(0, i)[2];
    }
    LG << flat_image.rows << " , " << flat_image.cols;
    LG << "-------------------";
    for (int i = 0; i < 30; i++) {
            LG << i << ": " << rgb_image.at<float>(0, i);
    }
    std::vector<float> data_buffer;
    for (int i = 0; i < flat_image.rows; ++i) {
        data_buffer.insert(data_buffer.end(), flat_image.ptr<float>(i), flat_image.ptr<float>(i) + flat_image.cols);
    }

    LOG(INFO) << "image loaded.";
    Symbol yolo3 = Symbol::Load("yolo3_darknet53_voc-symbol.json");
    auto data = Symbol::Variable("data");
    LG << "Net loaded";
    std::map<std::string, NDArray> all_params;
    std::map<std::string, NDArray> arg_params;
    std::map<std::string, NDArray> aux_params; 
    NDArray::Load("yolo3_darknet53_voc-0000.params", 0, &all_params);
    Context ctx = Context::cpu();
    std::string aux_prefix = "aux:";
    for (auto pp : all_params) {
        std::string full_name = pp.first;
        if (full_name.compare(0, aux_prefix.length(), aux_prefix) == 0) {
            // aux_params
            aux_params[full_name.substr(4)] = pp.second;
        } else {
            // else
            arg_params[full_name.substr(4)] = pp.second;
        }
    }

    for (auto args : arg_params) {
        LG << "args: " << args.first;
    }

    for (auto auxs: aux_params) {
        LG << "auxs: " << auxs.first;
    }

    arg_params["data"] = NDArray(data_buffer, Shape(1, rgb_image.rows, rgb_image.cols, 3), ctx);
    NDArray::WaitAll();
    LG << "params loaded, data: " << arg_params["data"].GetShape().size();
    // LG << "------------------";
    // for (int i = 0; i < 30; i++) {
    //     // LG << arg_params["data"].At(0, i, 0) << "," << arg_params["data"].At(0, i, 1) << "," <<  arg_params["data"].At(0, i, 2);
    //     // LG << arg_params["data"].At(0, 0, i) << "," << arg_params["data"].At(1, 0, i) << "," <<  arg_params["data"].At(2, 0, i);
    //     LG << arg_params["data"].GetData()[i];
    // }
    
    // return 0;
    // std::vector<NDArray> arg_arrays;
    // std::vector<NDArray> grad_arrays;
    // std::vector<OpReqType> grad_reqs;
    // std::vector<NDArray> aux_arrays;

    // yolo3.InferExecutorArrays(ctx, &arg_arrays, &grad_arrays, &grad_reqs,
    //                   &aux_arrays, arg_params, std::map<std::string, NDArray>(),
    //   std::map<std::string, OpReqType>(),
    //                   arg_params);

    // LG << "infered size: " << arg_arrays.size() << ", " << aux_arrays.size();

    // for (auto o : yolo3.GetInternals().ListOutputs()) {
    //     LG << o;
    // }
    // for (auto o : arg_params) {
    //     LG << o.first;
    // }
    // yolo3 = yolo3.GetInternals()["_defaultpreprocess0_init_mean"];
    
    
    Executor *exec = yolo3.SimpleBind(
      ctx, arg_params, std::map<std::string, NDArray>(),
      std::map<std::string, OpReqType>(), aux_params);

    LG << "exec binded.";

    exec->Forward(false);
    NDArray::WaitAll();
    LG << "infer finished.";
    auto ids = exec->outputs[0].Copy(Context(kCPU, 0));
    auto scores = exec->outputs[1].Copy(Context(kCPU, 0));
    auto bboxes = exec->outputs[2].Copy(Context(kCPU, 0));
    auto rshape = ids.GetShape();
    LG << "dim: " << rshape.size();
    for (int i = 0; i < rshape.size(); ++i) {
        LG << "dim " << i << " : " << rshape[i];
    }
    // LG << rshape.size() << " : " << rshape[0] << " , " << rshape[1] << " , " << rshape[2];
    NDArray::WaitAll();
    // LG << ids.GetData()[0] << ", " << ids.GetData()[1] << ", " << ids.GetData()[2];
    plot_bbox(image, bboxes, scores, ids, 0.5);
    // LG << arg_params["_defaultpreprocess0_init_mean"].GetData()[0];

    delete exec;
    MXNotifyShutdown();
    return 0;
}