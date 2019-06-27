<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Validate Your MXNet Installation

- [Python](#python)
- [Python with MKL](#python-with-mkl)
- [Python with GPU](#python-with-gpu)
- [Verify GPU training](#verify-gpu-training)


## Python

Start the python terminal.

```bash
$ python
```

Run a short MXNet Python program to create a 2X3 matrix of ones, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3))
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```

## Python with MKL

Instructions for validating MKL or MKLDNN can be found in the [MKLDNN_README](https://mxnet.incubator.apache.org/versions/master/tutorials/mkldnn/MKLDNN_README.html#verify-whether-mkl-works).

## Python with GPU

This is similar to the previous example, but this time we use `mx.gpu()``, to set MXNet's context to be GPU.

```python
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```


## Verify GPU Training

Clone the MXNet repository to download all of the MXNet examples.

```bash
git clone --recursive https://github.com/apache/incubator-mxnet.git mxnet
```

From the `mxnet` directory run the following:

```bash
python example/image-classification/train_mnist.py --network lenet --gpus 0
```
