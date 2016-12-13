from .vgg_16 import *
from .googlenet import *

VGG_16_AGE_1_112_112 = get_vgg16(input_shape=(1,112,112), n_classes=10)
VGG_16_AGE_3_112_112 = get_vgg16(input_shape=(3,112,112), n_classes=10)
VGG_16_GENDER_1_112_112 = get_vgg16(input_shape=(1,112,112), n_classes=2)
VGG_16_GENDER_3_112_112 = get_vgg16(input_shape=(3,112,112), n_classes=2)
GOOGLENET_AGE_3_224_224 = get_vgg16(input_shape=(3,224,224), n_classes=10)
GOOGLENET_GENDER_3_224_224 = get_vgg16(input_shape=(3,224,224), n_classes=2)