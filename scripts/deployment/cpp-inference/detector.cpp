#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

int main(int argc, char** argv) {
    LOG(INFO) << "begin";
    MXNotifyShutdown();
    return 0;
}