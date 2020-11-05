from gluoncv import data

class COCODetectionTiny(data.COCODetection):
    CLASSES = ['bicycle', 'motorcycle']

class COCOInstanceTiny(data.COCOInstance):
    CLASSES = ['bicycle', 'motorcycle']
