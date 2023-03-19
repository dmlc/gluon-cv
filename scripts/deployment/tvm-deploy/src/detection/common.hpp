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

using namespace cimg_library;

// resize image
inline CImg<float> ResizeShortWithin(CImg<float> & src, int short_size, int max_size, int mult_base) {
    src.resize(max_size, short_size, -100, -100, 3);
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

namespace viz {
// convert color from hsv to bgr for plotting
inline std::vector<unsigned char> HSV2BGR(float h, float s, float v) {
    CImg<float> color_f(1, 1, 1, 3, h, s, v);
    CImg<int> color_i = color_f.HSVtoRGB();
    int * value_i = color_i.data();
    unsigned char r = static_cast<unsigned char>(*value_i);
    unsigned char g = static_cast<unsigned char>(*(value_i+1));
    unsigned char b = static_cast<unsigned char>(*(value_i+2));
    std::vector<unsigned char> color_vec;
    color_vec.emplace_back(r);
    color_vec.emplace_back(g);
    color_vec.emplace_back(b);
    return color_vec;
}


// plot bounding boxes on raw image
inline CImg<float> PlotBbox(CImg<float> img, std::vector<float>& bboxes, std::vector<float>& scores, std::vector<float>& labels,
               float thresh, std::vector<std::string>& class_names,
               std::map<int, std::vector<unsigned char>> colors, bool verbose, std::string& str) {
    int num = scores.size();
    std::mt19937 eng;
    std::uniform_real_distribution<float> rng(0, 1);
    float hue = rng(eng);
    if (verbose) {
        LOG(INFO) << "Start Ploting with visualize score threshold: " << thresh;
    }
    std::stringstream ss;
    ss << "Object detection result:\n";
    for (int i = 0; i < num; ++i) {
        float score = scores[i];
        float label = labels[i];
        if (score < thresh) continue;
        if (label < 0) continue;

        int cls_id = static_cast<int>(label);
        if (colors.find(cls_id) == colors.end()) {
            // create a new color
            int csize = static_cast<int>(class_names.size());
            if (class_names.size() > 0) {
                float hue = label / csize;
                colors[cls_id] = HSV2BGR(hue * 255, 0.75, 0.95);
            } else {
                // generate color for this id
                hue += 0.618033988749895;  // golden ratio
                hue = fmod(hue, 1.0);
                colors[cls_id] = HSV2BGR(hue * 255, 0.75, 0.95);
            }
        }

        // draw bounding box
        auto color = colors[cls_id];
        img.draw_rectangle(bboxes[4*i], bboxes[4*i+1], bboxes[4*i+2], bboxes[4*i+3], color.data(),0.7f,~0U);


        if (verbose) {
            if (cls_id >= class_names.size()) {
                LOG(INFO) << "id: " << cls_id << ", scores: " << score;
            } else {
                LOG(INFO) << "id: " << class_names[cls_id] << ", scores: " << score;
            }

        }

        if (cls_id >= class_names.size()) {
            ss << "\tid: " <<  cls_id << ", scores: " << std::fixed << std::setprecision(3) << score << ", coordinates of bounding box: top_left-(" << bboxes[4*i] << " , " << bboxes[4*i+1] << "), bottom_right-(" << bboxes[4*i+2] << " , " << bboxes[4*i+3] << ")\n";
        }
        else {
            ss << "\tid: " <<  class_names[cls_id] << ", scores: " << std::fixed << std::setprecision(3) << score << ", coordinates of bounding box: top_left-(" << bboxes[4*i] << " , " << bboxes[4*i+1] << "), bottom_right-(" << bboxes[4*i+2] << " , " << bboxes[4*i+3] << ")\n";
        }



        // put text
        std::string txt;
        if (class_names.size() > cls_id) {
            txt += class_names[cls_id];
        }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3) << score;
        txt += " " + ss.str();
        // cv::putText(img, txt, cv::Point(pt1.x, pt1.y - 5), , 0.6, color, 1);
        img.draw_text(bboxes[4*i], bboxes[4*i+1],txt.c_str(),color.data(),0,0.7f,13);

    }
    str = ss.str();
    return img;
}
}  // namespace viz
