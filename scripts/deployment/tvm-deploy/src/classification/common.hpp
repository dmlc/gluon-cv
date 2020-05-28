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
 *  Copyright (c) 2020 by Contributors
 * \file common.hpp
 * \brief Common functions for GluonCV cpp inference demo
 * \author
 */
#define cimg_display 0
#define cimg_use_jpeg
#define cimg_use_png
#include "../CImg.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <map>
#include <cmath>
#include <random>
#include <iomanip>
#include <vector>

using namespace cimg_library;

// resize short and center crop
inline CImg<float> ResizeShortCrop(CImg<float> & src, int short_size) {
    double h = src.height();
    double w = src.width();
    double im_size_min = h;
    double im_size_max = w;
    if (w < h) {
        im_size_min = w;
        im_size_max = h;
    }
    double scale = static_cast<double>(short_size) / static_cast<double>(im_size_min);
    int new_w = static_cast<int>(std::floor(w * scale));
    int new_h = static_cast<int>(std::round(h * scale));

    src.resize(new_w, new_h, -100, -100, 3);

    int rec_x = static_cast<int>(std::round(new_w/2-112));
    int rec_y = static_cast<int>(std::round(new_h/2-112));

    src.crop(rec_x, rec_y, rec_x+223, rec_y+223);
    return src;
}

inline bool EndsWith(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline std::vector<std::string> LoadClassNames(std::string filename) {
    std::vector<std::string> classes;
    std::string line;
    std::ifstream infile(filename);
    while(std::getline(infile,line)) {
        size_t pos = line.find('\n');
        if (pos!=std::string::npos) {
            line.erase(pos, 1);
        }
        pos = line.find('\r');
        if (pos!=std::string::npos) {
            line.erase(pos, 1);
        }
        classes.emplace_back(line);
    }
    return classes;
}

namespace compute {
inline std::vector<int> Topk(std::vector<float>& output, int topk) {
    int num = output.size();
    std::map<float,int,std::greater<float>> map_sort;
    for (int i = 0; i < num; i++) {
        map_sort[output[i]] = i;
    }
    std::vector<int> ret(topk);
    std::map<float, int>::iterator iter = map_sort.begin();
    for (int i = 0; i < topk; i++) {
        ret[i] = (iter->second);
        ++iter;
    }
    return ret;
}

inline void Softmax(std::vector<float>& output, std::vector<float>& res) {
    int num = output.size();
    float max_v = output[0];
    for (int i = 1; i < num; i++) {
        if (output[i] > max_v)
            max_v = output[i];
    }
    float sum = 0;
    for (int i = 0; i < num; i++) {
        sum += std::exp(output[i]-max_v);
    }
    for (int i = 0; i < num; i++) {
        res[i] = std::exp(output[i]-max_v) / sum;
    }
}
} //namespace compute
