"""4. Computing FLOPS, latency and fps of a model
=======================================================

It is important to have an idea of how to measure a video model's speed, so that you can choose the model that suits best for your use case.
In this tutorial, we provide two simple scripts to help you compute (1) FLOPS, (2) number of parameters, (3) fps and (4) latency.
These four numbers will help you evaluate the speed of this model.
To be specific, FLOPS means floating point operations per second, and fps means frame per second.
In terms of comparison, (1) FLOPS, the lower the better,
(2) number of parameters, the lower the better,
(3) fps, the higher the better,
(4) latency, the lower the better.


In terms of input, we use the setting in each model's training config.
For example, I3D models will use 32 frames with stride 2 in crop size 224, but R2+1D models will use 16 frames with stride 2 in crop size 112.
This will make sure that the speed performance here correlates well with the reported accuracy number.
We list these four numbers and the models' accuracy on Kinetics400 dataset in the table below.


+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
|  Model                                 | FLOPS          | # params   | fps                 | Latency         | Top-1 Accuracy  |
+========================================+================+============+=====================+=================+=================+
| resnet18_v1b_kinetics400               | 1.819          | 11.382     | 264.01              | 0.0038          | 66.73           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| resnet34_v1b_kinetics400               | 3.671          | 21.49      | 151.96              | 0.0066          | 69.85           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| resnet50_v1b_kinetics400               | 4.110          | 24.328     | 114.05              | 0.0088          | 70.88           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| resnet101_v1b_kinetics400              | 7.833          | 43.320     | 59.56               | 0.0167          | 72.25           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| resnet152_v1b_kinetics400              | 11.558         | 58.963     | 36.93               | 0.0271          | 72.45           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_resnet50_v1_kinetics400            | 33.275         | 28.863     | 1719.50             | 0.0372          | 74.87           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_resnet101_v1_kinetics400           | 51.864         | 52.574     | 1137.74             | 0.0563          | 75.10           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_nl5_resnet50_v1_kinetics400        | 47.737         | 38.069     | 1403.16             | 0.0456          | 75.17           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_nl10_resnet50_v1_kinetics400       | 62.199         | 42.275     | 1200.69             | 0.0533          | 75.93           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_nl5_resnet101_v1_kinetics400       | 66.326         | 61.780     | 999.94              | 0.0640          | 75.81           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_nl10_resnet101_v1_kinetics400      | 80.788         | 70.985     | 890.33              | 0.0719          | 75.93           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_slow_resnet50_f8s8_kinetics400     | 41.919         | 32.454     | 1702.60             | 0.0376          | 74.41           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_slow_resnet50_f16s4_kinetics400    | 83.838         | 32.454     | 1406.00             | 0.0455          | 76.36           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_slow_resnet50_f32s2_kinetics400    | 167.675        | 32.454     | 860.74              | 0.0744          | 77.89           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_slow_resnet101_f8s8_kinetics400    | 85.675         | 60.359     | 1114.22             | 0.0574          | 76.15           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_slow_resnet101_f16s4_kinetics400   | 171.348        | 60.359     | 876.20              | 0.0730          | 77.11           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| i3d_slow_resnet101_f32s2_kinetics400   | 342.696        | 60.359     | 541.16              | 0.1183          | 78.57           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| r2plus1d_v1_resnet18_kinetics400       | 40.645         | 31.505     | 804.31              | 0.0398          | 71.72           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| r2plus1d_v1_resnet34_kinetics400       | 75.400         | 61.832     | 503.17              | 0.0636          | 72.63           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| r2plus1d_v1_resnet50_kinetics400       | 65.543         | 53.950     | 667.06              | 0.0480          | 74.92           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| r2plus1d_v2_resnet152_kinetics400      | 252.900        | 118.227    | 546.19              | 0.1172          | 81.34           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| ircsn_v2_resnet152_f32s2_kinetics400   | 74.758         | 29.704     | 435.77              | 0.1469          | 83.18           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| slowfast_4x16_resnet50_kinetics400     | 27.820         | 34.480     | 1396.45             | 0.0458          | 75.25           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| slowfast_8x8_resnet50_kinetics400      | 50.583         | 34.566     | 1297.24             | 0.0493          | 76.66           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| slowfast_8x8_resnet101_kinetics400     | 96.794         | 62.827     | 889.62              | 0.0719          | 76.95           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| tpn_resnet50_f8s8_kinetics400          | 50.457         | 71.800     | 1350.39             | 0.0474          | 77.04           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| tpn_resnet50_f16s4_kinetics400         | 99.929         | 71.800     | 1128.39             | 0.0567          | 77.33           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| tpn_resnet50_f32s2_kinetics400         | 198.874        | 71.800     | 716.89              | 0.0893          | 78.90           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| tpn_resnet101_f8s8_kinetics400         | 94.366         | 99.705     | 942.61              | 0.0679          | 78.10           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| tpn_resnet101_f16s4_kinetics400        | 187.594        | 99.705     | 754.00              | 0.0849          | 79.39           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+
| tpn_resnet101_f32s2_kinetics400        | 374.048        | 99.705     | 479.77              | 0.1334          | 79.70           |
+----------------------------------------+----------------+------------+---------------------+-----------------+-----------------+


.. note::

    Feel free to skip the tutorial because the speed computation scripts are self-complete and ready to launch.

    :download:`Download Full Python Script: get_flops.py<../../../scripts/action-recognition/get_flops.py>`

    :download:`Download Full Python Script: get_fps.py<../../../scripts/action-recognition/get_fps.py>`

    You can reproduce the numbers in the above table by

    ``python get_flops.py --config-file CONFIG`` and ``python get_fps.py --config-file CONFIG``

    If you encouter missing dependecy issue of ``thop``, please install the package first.

    ``pip install thop``

"""
