from keras.layers import *
from keras.models import Model
from .googlenet_custom_layers import PoolHelper, LRN
from keras.regularizers import l2
from keras import initializations

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

    conv1 = Convolution2D(96,7,7,subsample=(4,4),border_mode='valid', activation=activation,
        name='conv1_7x7',W_regularizer=l2(l2_reg))(input)

    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',
        name='pool1_3x3')(conv1)
    
    norm1 = LRN(name='norm1')(pool1)


    # ****************************************************************************
    # *                                  Conv 2                                  *
    # ****************************************************************************

    conv2 = Convolution2D(256,5,5,border_mode='valid', activation=activation,
        name='conv2_5x5', init=init, W_regularizer=l2(l2_reg))(norm1)
    
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool2_2x2')(conv2)
    
    norm2 = LRN(name='norm2')(pool2)


    # ****************************************************************************
    # *                                  Conv 3                                  *
    # ****************************************************************************
    
    conv3 = Convolution2D(384,3,3, border_mode='same', activation=activation,
        name='conv3_3x3', init=init, W_regularizer=l2(l2_reg))(norm2)
    
    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool3_2x2')(conv3)

    feats = Flatten(name='flatten')(pool3)


    # ****************************************************************************
    # *                                 Feature 1                                *
    # ****************************************************************************

    fc6 = Dense(512, activation=activation, init=init,
        name='fc6', W_regularizer=l2(l2_reg))(feats)
    
    drop_fc6 = Dropout(dropout, name='drop_fc6')(fc6)

    fc7 = Dense(512, activation=activation, init=init,
        name='fc7', W_regularizer=l2(l2_reg))(drop_fc6)
    
    drop_fc7 = Dropout(dropout, name='drop_fc7')(fc7)

    prob = Dense(n_classes, activation='softmax', init=init,
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
        name='conv1_7x7',W_regularizer=l2(l2_reg))(input)

    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',
        name='pool1_3x3')(conv1)
    
    norm1 = LRN(name='norm1')(pool1)


    # ****************************************************************************
    # *                                  Conv 2                                  *
    # ****************************************************************************

    conv2 = Convolution2D(256,5,5, border_mode='valid', activation=activation,
        name='conv2_5x5',W_regularizer=l2(l2_reg))(norm1)
    
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool2_2x2')(conv2)
    
    norm2 = LRN(name='norm2')(pool2)


    # ****************************************************************************
    # *                                  Conv 3                                  *
    # ****************************************************************************
    
    conv3 = Convolution2D(384,3,3, border_mode='same', activation=activation,
        name='conv3_3x3',W_regularizer=l2(l2_reg))(norm2)
     
    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid',
        name='pool3_2x2')(conv3)

    feats = Flatten(name='flatten')(pool3)


    # ****************************************************************************
    # *                                 Feature 1                                *
    # ****************************************************************************

    l1fc6 = Dense(1024, activation=activation,
        name='loss1_fc6', W_regularizer=l2(l2_reg))(feats)
    
    l1_drop_fc6 = Dropout(dropout, name='loss1_drop_fc6')(l1fc6)

    l1fc7 = Dense(1024, activation=activation,
        name='loss1_fc7', W_regularizer=l2(l2_reg))(l1_drop_fc6)
    
    l1_drop_fc7 = Dropout(dropout, name='loss1_drop_fc7')(l1fc7)

    prob1 = Dense(n_classes[0], activation='softmax',
        name='prob1', W_regularizer=l2(l2_reg))(l1_drop_fc7)
    

    # ****************************************************************************
    # *                                 Feature 2                                *
    # ****************************************************************************

    l2fc6 = Dense(512, activation=activation,
        name='loss2_fc6', W_regularizer=l2(l2_reg))(feats)
    
    l2_drop_fc6 = Dropout(dropout, name='loss2_drop_fc6')(l2fc6)

    l2fc7 = Dense(512, activation=activation,
        name='loss2_fc7', W_regularizer=l2(l2_reg))(l2_drop_fc6)
    
    l2_drop_fc7 = Dropout(dropout, name='loss2_drop_fc7')(l2fc7)

    prob2 = Dense(n_classes[1], activation='softmax',
        name='prob2', W_regularizer=l2(l2_reg))(l2_drop_fc7)


    # ****************************************************************************
    # *                                   Model                                  *
    # ****************************************************************************
    
    model = Model(input=input, output=[prob1, prob2])
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model
