from gluoncv import data

class COCODetectionMini(data.COCODetection):
    CLASSES = ['bicycle', 'motorcycle']

class COCOInstanceMini(data.COCOInstance):
    CLASSES = ['bicycle', 'motorcycle']
