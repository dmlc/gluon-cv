import os

def main():
    model_list_c = ['ResNet18_v1','ResNet34_v1','ResNet50_v1','ResNet101_v1','ResNet152_v1','ResNet18_v2','ResNet34_v2',\
              'ResNet50_v2','ResNet101_v2','ResNet152_v2','ResNext50_32x4d','ResNext101_32x4d',\
              'SE_ResNext50_32x4d','SE_ResNext101_32x4d','SE_ResNext101_64x4d','VGG11_bn','VGG13_bn','VGG16_bn','VGG19_bn',\
             'SqueezeNet1.0','SqueezeNet1.1','DenseNet121','DenseNet161','DenseNet169','DenseNet201','MobileNet1.0',\
              'MobileNetV2_1.0','MobileNetV3_Large','MobileNetV3_Small','MobileNet0.5',\
             'MobileNet0.25','MobileNetV2_1.0','MobileNetV2_0.75','MobileNetV2_0.5','MobileNetV2_0.25']
    name_prev = "gcv_lite_classify"
    for model_name in model_list_c:
        cmd_r = "mv {} {}".format(name_prev, model_name)
        os.system(cmd_r)
        cmd_m = "python ../tools/export_classify.py --model {}".format(model_name)
        os.system(cmd_m)
        cmd_p = "zip {}.zip {} model_0000.params model_graph.json model_lib.so imagenet_classes.txt".format(model_name, model_name)
        os.system(cmd_p)
        name_prev = model_name
        
    model_list_d = ['ssd_300_vgg16_atrous_voc','ssd_512_vgg16_atrous_voc','ssd_512_resnet50_v1_voc','ssd_512_mobilenet1.0_voc',\
              'faster_rcnn_resnet50_v1b_voc','yolo3_darknet53_voc','yolo3_mobilenet1.0_voc']
    name_prev = "gcv_lite_detection"
    for model_name in model_list_d:
        cmd_r = "mv {} {}".format(name_prev, model_name)
        os.system(cmd_r)
        cmd_m = "python ../tools/export_detect.py --model {}".format(model_name)
        os.system(cmd_m)
        cmd_p = "zip {}.zip {} model_0000.params model_graph.json model_lib.so".format(model_name, model_name)
        os.system(cmd_p)
        name_prev = model_name
    
    
if __name__ == "__main__":
    main()
