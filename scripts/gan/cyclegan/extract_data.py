from gluoncv.utils import download
import zipfile
download('https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/maps.zip', './datasets/')
zip_ref = zipfile.ZipFile('./datasets/maps.zip', 'r')
zip_ref.extractall('./datasets')
zip_ref.close()
