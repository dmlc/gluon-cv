# Export GlonCV Model to ONNX and Inference with ONNX Runtime

### GluonCV provides ready to use ONNX model. Here is a list:

[Exported Lists](./exported_models.csv)

### Inference with ONNX model

Inference with ONNX model is straightforward:

1. Prepare the input data. 

   All 2-D models contain preprocess layers, so you don't need to preprocess it. However, you still need to resize it to match the input shape. For example:

   ```python
   test_img_file = os.path.join('test_imgs', 'bikers.jpg')
   img = mx.image.imread(test_img_file)
   img = mx.image.imresize(img, 224, 224)
   img = img.expand_dims(0).astype('float32')
   ```

2. Get an ONNX inference session and its input name. For example:

   ```python
   onnx_session = onnxruntime.InferenceSession(onnx_model, None)
   input_name = onnx_session.get_inputs()[0].name
   ```

3. Make the prediction. For example:

   ```python
   onnx_result = onnx_session.run([], {input_name: img.asnumpy()})[0]
   ```

### Export GluonCV Models to ONNX

For those who are interested, below is a guide on how to export the model yourself:

To be notice, current released version of mxnet(1.8.0) does not support many operators. Therefore, in order to export models to onnx, you'll have to either

1. Build the newest version of mxnet from source
2. Checkout the `installation` section [here](https://github.com/apache/incubator-mxnet/tree/v1.x/python/mxnet/onnx)

There are three main steps to export the model:

1. Get the GluonCV model using `gluoncv.model_zoo.get_model`
2. Export the GluonCV model to get the symbol and parameter files using `export_block`
3. Export the ONNX model with symbol and parameter files using `mxnet.contrib.onnx.export_model`

Here is a snippet of example code. You won't be able to run it directly, but it can give you some idea on how to export on your own.

```python
model_type = {'0': 'Obj Classification',
              '1': 'Obj Detection',
              '2': 'Img Segmentation',
              '3': 'Pose Estimation',
              '4': 'Action Recognition',
              '5': 'Depth Prediction'}

class Model():
    
    def __init__(self, model_name, input_shape, model_type):
        self.model_name = model_name
        self.model_type = model_type
        if len(input_shape) == 4:
            input_shape.append(input_shape.pop(1)) # change BCHW to BHWC
        self.input_shape = tuple(input_shape)
        self.ctx = mx.cpu(0)
        
        self.param_path = 'params'
        self.exported_model_path = 'exported_models'
        self.exported_model_prefix = 'gluoncv_exported_'
        self.exported_model_suffix = '.onnx'
        self.symbol_suffix = '-symbol.json'
        self.params_suffix = '-0000.params'
        self.onnx_model = os.path.join(self.exported_model_path, 
                                      self.exported_model_prefix+self.model_name+self.exported_model_suffix)
        
    def exists(self):
        return os.path.isfile(self.onnx_model)
    
    def is_3D(self):
        return len(self.input_shape) == 5
        
    def get_model(self):
        self.gluon_model = gcv.model_zoo.get_model(self.model_name, pretrained=True, ctx=self.ctx)
    
    def export(self):
        sym = os.path.join(self.param_path, self.model_name + self.symbol_suffix)
        params = os.path.join(self.param_path, self.model_name + self.params_suffix)
        if self.is_3D():
            export_block(os.path.join(self.param_path,self.model_name), self.gluon_model, 
                         data_shape=self.input_shape[1:], preprocess=False, layout='CTHW')
        else:
            export_block(os.path.join(self.param_path,self.model_name), self.gluon_model, 
                         data_shape=self.input_shape[1:], preprocess=True, layout='HWC')
            
    def export_onnx(self):
        sym = os.path.join(self.param_path, self.model_name + self.symbol_suffix)
        params = os.path.join(self.param_path, self.model_name + self.params_suffix)
        return onnx_mxnet.export_model(sym, params, [self.input_shape], np.float32, self.onnx_model)
      
model = Model('resnet18_v1', [1,3,224,224], '0')
model.get_model()
model.export()
model.export_onnx()
```