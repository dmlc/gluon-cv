/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file detection.cpp
 */
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_runtime_api.h>
#include "common.hpp"
#include "clipp.hpp"
#include <chrono>

namespace synset {
// some commonly used datasets
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

// by default class names are empty
static std::vector<std::string> CLASS_NAMES = {};
}  // namespace synset

namespace args {
static std::string model = "yolo3_darknet53_voc";
static std::string image;
static std::string output_file;
static std::string output_image;
static std::string class_name_file;
static int epoch = 0;
static int gpu = -1;
static bool quite = false;
static bool no_display = true;
static float viz_thresh = 0.3;
static int min_size = 512;
static int max_size = 605;
static int multiplier = 32;  // just to ensure image shapes are multipliers of feature strides, for yolo3 models
}  // namespace args

void ParseArgs(int argc, char** argv) {
    using namespace clipp;

    auto cli = (
        value("image file", args::image),
        (option("-f", "--output-file") & value(match::prefix_not("-"), "outfile", args::output_file)) % "output file, by default no output",
        (option("-i", "--output-image") & value(match::prefix_not("-"), "outimage", args::output_image)) % "output image, by default no output",
        (option("--class-file") & value(match::prefix_not("-"), "classfile", args::class_name_file)) % "plain text file for class names, one name per line",
        (option("-e", "--epoch") & integer("epoch", args::epoch)) % "Epoch number to load parameters, by default is 0",
        (option("--gpu") & integer("gpu", args::gpu)) % "Which gpu to use, by default is -1, means cpu only.",
        option("-q", "--quite").set(args::quite).doc("Quite mode, no screen output"),
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
            LOG(ERROR) << "Cannot determine class names, you can specify --class-file with a text file...";
        }
    } else {
        synset::CLASS_NAMES = LoadClassNames(args::class_name_file);
    }
}

void RunDemo() {
  // tvm module for compiled functions
  tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile("model_lib.so");
  std::ifstream json_in("model_graph.json", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  std::ifstream params_in("model_0000.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;

  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

  // load image data
  CImg<float> image(args::image.c_str());
  // resize to match the input of model
  image = ResizeShortWithin(image, args::min_size, args::max_size, args::multiplier);
    
  DLTensor* input;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, image.height(), image.width(), 3};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
  
  TVMArrayCopyFromBytes(input,image.data(),image.height()*image.width()*3*4);

  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("data", input);

  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  auto start = std::chrono::steady_clock::now();
  run();
  
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  tvm::runtime::NDArray ids = get_output(0);
  tvm::runtime::NDArray scores = get_output(1);
  tvm::runtime::NDArray bboxes = get_output(2);
  auto end = std::chrono::steady_clock::now();
    if (!args::quite) {
        LOG(INFO) << "Elapsed time {Forward->Result}: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms";
    }
  int num = bboxes.Shape()[1];
  std::vector<float> id_vec(num);
  std::vector<float> score_vec(num);
  std::vector<float> bbox_vec(num * 4);
  memcpy(id_vec.data(),ids->data,num*4);
  memcpy(score_vec.data(),scores->data,num*4);
  memcpy(bbox_vec.data(),bboxes->data,num*4*4);
  
  // draw boxes
  std::string str_res;
  auto plt = viz::PlotBbox(image, bbox_vec, score_vec, id_vec, args::viz_thresh, synset::CLASS_NAMES, std::map<int, std::vector<unsigned char>>(), !args::quite, str_res);

  // display drawn image
    if (!args::no_display) {
        
    }

    // output image
    if (!args::output_image.empty()) {
        plt.save(args::output_image.c_str());
    }
    
    //output file
    if (!args::output_file.empty()) {
        std::ofstream outfile(args::output_file);
        outfile << str_res;
        outfile.close();
    }


  TVMArrayFree(input);
}

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    RunDemo();
    return 0;
}
