from keras.layers import *
from keras.models import Model
from .googlenet_custom_layers import PoolHelper, LRN
from keras.regularizers import l2

def get_levinet(params, weights_path=None):
    
    input_shape = params.get('input_shape', (3,224,224))
    activation = params.get('activation', 'relu')
    l2_reg = params.get('l2_reg', 2e-4)
    dropout = params.get('dropout', 0.4)
    n_classes = params.get('n_classes', 10)
    init = params.get('init', 'glorot_uniform')

    input = Input(shape=input_shape)
    

    # ****************************************************************************
    # *                                  Conv 1                                  *
    # ****************************************************************************

    conv1 = Convolution2D(96,7,7,subsample=(2,2),border_mode='valid', activation=activation,
        name='conv1/7x7',W_regularizer=l2(l2_reg))(input)

    pool1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='valid',
        name='pool1/2x2')(conv1)
    
    norm1 = LRN(name='norm1')(pool1)


    # ****************************************************************************
    # *                                  Conv 2                                  *
    # ****************************************************************************

    conv2 = Convolution2D(256,5,5, border_mode='same', activation=activation,
        name='conv2/5x5',W_regularizer=l2(l2_reg))(norm1)
    
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool2/2x2')(conv2)
    
    norm2 = LRN(name='norm2')(pool2)


    # ****************************************************************************
    # *                                  Conv 3                                  *
    # ****************************************************************************
    
    conv3 = Convolution2D(384,3,3, border_mode='same', activation=activation,
        name='conv3/3x3',W_regularizer=l2(l2_reg))(norm2)
    
    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool3/2x2')(conv3)

    feats = Flatten(name='flatten')(pool3)


    # ****************************************************************************
    # *                                 Feature 1                                *
    # ****************************************************************************

    fc6 = Dense(512, activation=activation,
        name='loss/fc6', W_regularizer=l2(l2_reg))(feats)
    
    drop_fc6 = Dropout(dropout, name='loss/drop/fc6')(fc6)

    fc7 = Dense(512, activation=activation,
        name='loss/fc7', W_regularizer=l2(l2_reg))(drop_fc6)
    
    drop_fc7 = Dropout(dropout, name='loss/drop/fc7')(fc7)

    prob = Dense(n_classes, activation=activation,
        name='prob', W_regularizer=l2(l2_reg))(drop_fc7)
    

    # ****************************************************************************
    # *                                   Model                                  *
    # ****************************************************************************
    
    model = Model(input=input, output=prob)
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model


def get_levinet_multi(params, weights_path=None):
    
    input_shape = params.get('input_shape', (3,224,224))
    activation = params.get('activation', 'relu')
    l2_reg = params.get('l2_reg', 2e-4)
    dropout = params.get('dropout', 0.4)
    n_classes = params.get('n_classes', [10, 2])
    init = params.get('init', 'glorot_uniform')

    input = Input(shape=input_shape)
    

    # ****************************************************************************
    # *                                  Conv 1                                  *
    # ****************************************************************************

    conv1 = Convolution2D(96,7,7,subsample=(2,2),border_mode='valid', activation=activation,
        name='conv1/7x7',W_regularizer=l2(l2_reg))(input)

    pool1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='valid',
        name='pool1/2x2')(conv1)
    
    norm1 = LRN(name='norm1')(pool1)


    # ****************************************************************************
    # *                                  Conv 2                                  *
    # ****************************************************************************

    conv2 = Convolution2D(256,5,5, border_mode='same', activation=activation,
        name='conv2/5x5',W_regularizer=l2(l2_reg))(norm1)
    
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool2/2x2')(conv2)
    
    norm2 = LRN(name='norm2')(pool2)


    # ****************************************************************************
    # *                                  Conv 3                                  *
    # ****************************************************************************
    
    conv3 = Convolution2D(384,3,3, border_mode='same', activation=activation,
        name='conv3/3x3',W_regularizer=l2(l2_reg))(norm2)
    
    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool3/2x2')(conv3)

    feats = Flatten(name='flatten')(pool3)


    # ****************************************************************************
    # *                                 Feature 1                                *
    # ****************************************************************************

    l1fc6 = Dense(512, activation=activation,
        name='loss1/fc6', W_regularizer=l2(l2_reg))(feats)
    
    l1_drop_fc6 = Dropout(dropout, name='loss1/drop/fc6')(l1fc6)

    l1fc7 = Dense(512, activation=activation,
        name='loss1/fc7', W_regularizer=l2(l2_reg))(l1_drop_fc6)
    
    l1_drop_fc7 = Dropout(dropout, name='loss1/drop/fc7')(l1fc7)

    prob1 = Dense(n_classes[0], activation=activation,
        name='prob1', W_regularizer=l2(l2_reg))(l1_drop_fc7)
    

    # ****************************************************************************
    # *                                 Feature 2                                *
    # ****************************************************************************

    l2fc6 = Dense(512, activation=activation,
        name='loss2/fc6', W_regularizer=l2(l2_reg))(feats)
    
    l2_drop_fc6 = Dropout(dropout, name='loss2/drop/fc6')(l2fc6)

    l2fc7 = Dense(512, activation=activation,
        name='loss2/fc7', W_regularizer=l2(l2_reg))(l2_drop_fc6)
    
    l2_drop_fc7 = Dropout(dropout, name='loss2/drop/fc7')(l2fc7)

    prob2 = Dense(n_classes[1], activation=activation,
        name='prob2', W_regularizer=l2(l2_reg))(l2_drop_fc7)


    # ****************************************************************************
    # *                                   Model                                  *
    # ****************************************************************************
    
    model = Model(input=input, output=[prob1, prob2])
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model
