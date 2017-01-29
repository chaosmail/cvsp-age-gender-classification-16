import time
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.optimizers import SGD
from models.googlenet_custom_layers import PoolHelper, LRN

from transformation import *

ageAndGenderModel = None  # global variable that holds the loaded model


def classify_image(image, model=None, means=[0.45008409, 0.37675238, 0.3356632], stds=[0.28836954, 0.26367465, 0.2598381]):
    """
    Classify an image with the age and gender classifier. Before calling this method, please set the model with
    load_classifier() or set_classifier() or you specifiy a model to this method by setting the model parameter.

    Parameters:
    image: An image in a typical numpy standard format. We tested it for example with images generated by:
        with Image.open(image_path) as img:
            img = np.asarray(img)
        The dimensions should be 112x112x3 and the face should be visible and if possible at least a little bit
        centered and big enough to be able to recognise something. So a face cropping + registration will improve
        the results.
    model: Optional if already set. If you want to use another model than set by load_classifier() or
        set_classifier() you can use this parameter. Otherwise please use the two mentioned functions before to set the
        model once. Not every model can be used because this method expects a model that outputs 2 seperate results for
        gender and age. If you want to use a model that does not fulfill this criteria, you can easily change the code.
    means: Optional. The mean which is subtracted from the images. Default values were computed from the
        IMDB-Wiki dataset
    stds: Optional. The standard deviation which is used to normalize the image. Default values were computed
        from the IMDB-Wiki dataset

    Returns: a list with 4 values: [gender_class_id, age_class_id, gender_class_confidence, age_class_confidence].
    The gender_class_id is 0 or 1 and represents an index for the following classes:
    ['female', 'male']
    The age_class_id is an index of the following array of age classes:
    ['[0 - 15]', '[16 - 20]', '[21 - 25]', '[26 - 30]', '[31 - 35]',
     '[36 - 40]', '[41 - 45]', '[46 - 50]', '[51 - 55]', '[56 - 100]']


    General Description / How-To:
    [0. Install needed libraries if necessary. Numpy, Keras, Theano/Tensorflow (we used Theano),...]
    1. Load this source file by using something like:
        'import age_gender_classify', or 'from age_gender_classify import classify_image, load_classifier'
    2. Load the classifier with load_classifier() or if you already loaded it somewhere else (probably not) you can use
        set_classifier(). Do this only once at the beginning (ex. server startup) as it is costly.
    3. As soon as you get an image (if it has to be loaded, you can do this by calling load_image() but you will
        probably have it already in some format, if received by a server) you can classify it with classify_image().

    If these instructions are not clear enough, look at the method demo_classification() where a complete
    classification is performed (except for the imports as it is in the same file).
    """
    if model is None:
        if ageAndGenderModel is None:
            print('ERROR: Please load an age and gender model before trying to classify an image with it...')
            exit()
        model = ageAndGenderModel  # take the standard model because nothing is passed

    sample = preprocess_image(image, means=means, stds=stds, dims=(112, 112, 3), out_shape=(3, 112, 112))

    # Expand the sample (3,112,112) to batch dimension (1,3,112,112)
    X = np.expand_dims(sample, axis=0)

    #probas = model.predict_proba(X, batch_size=1, verbose=0).reshape(-1)
    predictions = model.predict(X, batch_size=1, verbose=0)
    #predictions_classes = model.predict_classes(X, batch_size=1, verbose=0)


    # Get the class index of the highest probability
    gender_class_id = np.argmax(predictions[1][0])
    gender_class_confidence = predictions[1][0][gender_class_id]
    age_class_id = np.argmax(predictions[0][0])
    age_class_confidence = predictions[0][0][age_class_id]

    return [gender_class_id, age_class_id, gender_class_confidence, age_class_confidence]



def load_classifier(model_path='ageGenderModel.h5'):
    """
    Load the classifier specified by the path. This has to be done once before images can be classified and should
    not be done more than once because it can take a few seconds.

    Parameters:
        model_path: the path to the h5 model file. Default is the ageGenderModel.h5 file that should be on the
        same folder level than the script

    Returns: the model. But it is unnecessary to use the returned model object because the model is also saved in
    a variable which is then used by the classify_image() method. But if you want to use several different model files
    you can use the returned model and pass it as an optional parameter to classify_image()
    """
    try:
        model = load_model(model_path, custom_objects={'LRN':LRN})
        # print(" Input shape: %s, %i classes" % (
        #   str(model.layers[0].batch_input_shape[1:]), model.layers[-1].output_dim))
        learning_rate = 1e-2
        opt = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        global ageAndGenderModel
        ageAndGenderModel = model
        return model
    except OSError as e:
        print('Failed loading model %s' % model_path)
        print(e)
        exit()


def set_classifier(model):
    """
    Set the standard model manually. Will probably not be necessary as it is set automatically by load_classifier

    Parameters:
        model: the new standard model

    Returns: nothing
    """
    global ageAndGenderModel
    ageAndGenderModel = model


def preprocess_image(sample, means=None, stds=None, dims=None, out_shape=None):
    """
    Internally used method for preprocessing an image which should be classified.
    The image is: resized, casted to float, the mean is subtracted, it is divided by the std and the
    dimensions are changed to be supported by Keras (112x112x3 --> 3x112x112)

    Parameters:
        sample: the image to preprocess
        means: the means to subtract from the image
        stds: the standard deviation which is used to normalize the image
        dims: the image dimensions (should be 112x112x3)
        out_shape: the image dimensions after reshaping (should be 3x112x112)

    Returns: the preprocessed image
    """

    # print("Preprocessing image ...")
    # print(" Transformations in order:")
    tform = TransformationSequence()

    if dims is not None:
        t = ResizeTransformation(dims)
        tform.add_transformation(t)
        # print("  %s" % t.__class__.__name__)

    t = FloatCastTransformation()
    tform.add_transformation(t)
    # print("  %s" % t.__class__.__name__)

    if means is not None:
        t = PerChannelSubtractionImageTransformation(np.array(means, dtype=np.float32))
        tform.add_transformation(t)
        # print("  %s (%s)" % (t.__class__.__name__, str(t.values())))

    if stds is not None:
        t = PerChannelDivisionImageTransformation(np.array(stds, dtype=np.float32))
        tform.add_transformation(t)
        # print("  %s (%s)" % (t.__class__.__name__, str(t.values())))

    if out_shape is not None:
        t = ReshapeTransformation(out_shape)
        tform.add_transformation(t)
        # print("  %s %s" % (t.__class__.__name__, str(t.value())))

    sample = tform.apply(sample)
    # print(" Result: shape: %s, dtype: %s, mean: %.3f, std: %.3f" % (
    #   sample.shape, sample.dtype, sample.mean(), sample.std()))

    return sample



def demo_classification():
    """
    Performs a demo classification. Look into the code if you need help with using the above functions.
    Example output on a relatively slow machine with no GPU support:
    Starting Demo Classification...
    Loading of the model took 12.141696214675903 seconds
    The person on the image is male and [21 - 25] years old
    Classification took 0.45056692361831663 seconds per image
    """

    print('Starting Demo Classification...')

    # 1. Load model
    start = time.time()
    model = load_classifier('ageGenderModel.h5')
    end = time.time()
    print('Loading of the model took ' + str(end-start) + ' seconds')

    # 2. Load Images (or get them somewhere)
    image = load_image('../data/obama112.png')

    # 3. Classify Images
    start = time.time()
    dict_gender = ['female', 'male']
    dict_age = [
        '[0 - 15]', '[16 - 20]', '[21 - 25]', '[26 - 30]', '[31 - 35]',
        '[36 - 40]', '[41 - 45]', '[46 - 50]', '[51 - 55]', '[56 - 100]'
    ]
    iterations = 20  # Try several times to see how long it really takes
    for i in range(0, iterations):
        classification_results = classify_image(image)
        gender_class_id = classification_results[0]
        age_class_id = classification_results[1]
        gender_class_confidence = classification_results[2]
        age_class_confidence = classification_results[3]
    end = time.time()
    # Take last result as an example and print it:
    print('The person on the image is ' + dict_gender[gender_class_id] + ' and ' + dict_age[age_class_id] + ' years old')

    # Note that if you want to classify more images, Keras offers also batch classification
    print('Classification took ' + str((end-start)/iterations) + ' seconds per image')



def load_image(image_path):
    """
    Loads an image specified by the path from the disc.
    """

    # print("Loading image ...")
    try:
        with Image.open(image_path) as img:
            img_data = np.asarray(img)
            # print(" Shape: %s" % str(img_data.shape))
    except FileNotFoundError as e:
        print('Failed loading image %s' % image_path)
        exit()
    return img_data
