# GluonCV lite models
This is a demo application which illustrates how to use pretrained GluonCV models in c++ environments with the support of TVM.

## Build instruction
Please clone the full TVM repository by `git clone --recursive https://github.com/apache/incubator-tvm.git tvm`

Since we want to build libjpeg and libpng statically and link to them in order that the executables could work on user's clean environment, you cannot directly build using the existing files.

You need to statically build Zlib, libjpeg and libpng, the outputs should be libz.a, libpng.a and libjpeg.a respectively. I didn't provide these files because they depend on specific operating system. Remember to add the path of the three output `*.a` files using `link_directories()` in CMakeLists.txt. During my building process, I put those three files in the `build` directory under current path which correspond to the command of line 17 in CMakeLists.txt.

Also remember to add path of the libraries' header files using `INCLUDE_DIRECTORIES()` in CMakeLists.txt.

After doing things above, you can build with the following commands:


```
mkdir -p build && cd build
cmake ..
make
```

I also write a script to automatically pack binary with different models in the pack folder.

Some prebuilt binaries are uploaded to [this website](https://zyliu-cv.s3-us-west-2.amazonaws.com/gluoncv-lite/index.html).

Usage of image classification models:
```
SYNOPSIS
        ./<model name> <image file> [-o <outfile>] [--class-file <classfile>] 
                         [--topk <topk>]

OPTIONS
        -o, --output <outfile>
                    output file for text results

        --class-file <classfile>
                    plain text file for class names, one name per line

        --topk <topk> number of the most probable classes to show as output
```

Usage of object detection models:
```
SYNOPSIS
        ./<model name> <image file> [-f <outfile>]  [-i <outimagr>]  
                         [--class-file <classfile>] [-t <thresh>]

OPTIONS
        -f, <outfile>
                    output text file
                    
        -i, <outimage>
                    output image

        --class-file <classfile>
                    plain text file for class names, one name per line

        -t, --thresh <thresh>
                    Visualize threshold, from 0 to 1, default 0.3.
```
