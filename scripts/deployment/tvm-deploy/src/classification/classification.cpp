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
 * \brief code to deploy image classification models
 * \file classification.cpp
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
// by default class names are empty
static std::vector<std::string> CLASS_NAMES = {};
}  // namespace synset

namespace args {
static std::string image;
static std::string output;
static std::string class_name_file;
static int epoch = 0;
static int gpu = -1;
static int topk = 5;
static bool quite = false;
static bool no_display = false;
static int min_size = 256;
}  // namespace args

void ParseArgs(int argc, char** argv) {
    using namespace clipp;

    auto cli = (
        value("image file", args::image),
        (option("-o", "--output") & value(match::prefix_not("-"), "outfile", args::output)) % "output file, by default no output",
        (option("--class-file") & value(match::prefix_not("-"), "classfile", args::class_name_file)) % "plain text file for class names, one name per line",
        (option("-e", "--epoch") & integer("epoch", args::epoch)) % "Epoch number to load parameters, by default is 0",
        (option("--gpu") & integer("gpu", args::gpu)) % "Which gpu to use, by default is -1, means cpu only.",
        (option("--topk") & integer("topk", args::topk)) % "Number of the most probable classes to show as output, by default is 5",
        option("-q", "--quite").set(args::quite).doc("Quite mode, no screen output"),
        option("--no-disp").set(args::no_display).doc("Do not display result")
    );
    if (!parse(argc, argv, cli) || args::image.empty()) {
        std::cout << make_man_page(cli, argv[0]);
        exit(-1);
    }

    // parse class names
    if (args::class_name_file.empty()) {
        synset::CLASS_NAMES = LoadClassNames("imagenet_classes.txt");
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

  DLTensor* input;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, 3, 224, 224};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
  // load image data
  
  CImg<float> image(args::image.c_str());
  
  
  image = ResizeShortCrop(image, args::min_size);
  
  
  TVMArrayCopyFromBytes(input,image.data(),224*224*3*4);

  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("data", input);

  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  auto start = std::chrono::steady_clock::now();
  run();
  

  DLTensor* y;
  int out_ndim = 2;
  int64_t out_shape[2] = {1, 1000};
  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

  // get the function from the module(get output data)
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  get_output(0, y);
  auto end = std::chrono::steady_clock::now();
    if (!args::quite) {
        LOG(INFO) << "Elapsed time {Forward->Result}: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms";
    }
    
  std::vector<float> output(1000);
  TVMArrayCopyToBytes(y, output.data(), 1000 * 4);
  std::vector<int> res_topk = compute::Topk(output, args::topk);
  
  std::vector<float> res_softmax(1000);
  compute::Softmax(output, res_softmax);
  std::stringstream ss;
  ss << "The input picture is classified to be\n";
  for (int i = 0; i < args::topk; i++) {
      ss <<"\t[" << synset::CLASS_NAMES[res_topk[i]] <<  "], with probability " << std::fixed << std::setprecision(3) << res_softmax[res_topk[i]] << "\n";
  }

  // display result
  if (!args::no_display) {
      std::cout << ss.str();
  }

  // output file
  if (!args::output.empty()) {
      std::ofstream outfile(args::output);
      outfile << ss.str();
      outfile.close();
  }

  TVMArrayFree(input);
  TVMArrayFree(y);
}

int main(int argc, char** argv) {
    ParseArgs(argc, argv);
    RunDemo();
    return 0;
}
