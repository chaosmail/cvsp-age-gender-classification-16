from keras.utils.np_utils import to_categorical
import json
settings = json.load(open("config", "r"))
path_to_save_debug_convoluted_images_to = settings["path_to_save_debug_convoluted_images_to"]


def evaluate_with_keras_method(model, data):
    print("Evaluating with test data...")
    accuracy = model.evaluate(data.test_images[:, :, :], data.test_labels_cat[:, :], batch_size=32)
    print("Results:")
    # print("Loss: "+str(loss))
    print("Accuracy: "+str(accuracy))
    print("Finished evaluation")


def evaluate_manually(model, data):
    print("Evaluating with test data...")
    predicted_classes = model.predict_classes(data.test_images, batch_size=32)
    correctly_classified = 0
    falsely_classified = 0
    for i in range(0, predicted_classes.shape[0]-1):
        if predicted_classes[i] == data.test_labels[i, 0]:
            correctly_classified += 1
        else:
            falsely_classified += 1
    accuracy = correctly_classified / (correctly_classified+falsely_classified)
    print("Correct: "+str(correctly_classified))
    print("False: "+str(falsely_classified))
    print("Accuracy: "+str(accuracy))


def evaluate_manually_with_training_data(model, data):
    print("Evaluating with training data...")
    predicted_classes = model.predict_classes(data.train_images[0:6000], batch_size=32)  # TODO: test with all!
    correctly_classified = 0
    falsely_classified = 0
    for i in range(0, predicted_classes.shape[0]-1):
        if predicted_classes[i] == data.train_labels[i, 0]:
            correctly_classified += 1
        else:
            falsely_classified += 1
    accuracy = correctly_classified / (correctly_classified+falsely_classified)
    print("Correct: "+str(correctly_classified))
    print("False: "+str(falsely_classified))
    print("Accuracy: "+str(accuracy))


def plot_model(model):
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True)

    print("Plotted the model to model.png")


def show_layer(model, layer_number, input_image):
    # http://keras.io/getting-started/faq/#how-can-i-visualize-the-output-of-an-intermediate-layer
    from keras import backend as k
    layer_output = k.function([model.layers[0].input, k.learning_phase()],
                                      [model.layers[layer_number].output])

    # output in test mode = 0
    # layer_output_test_mode = layer_output([input_image, 0])[0]

    # output in train mode = 1
    layer_output_train_mode = layer_output([input_image, 1])[0]

    resulted_image = layer_output_train_mode

    # Create directory or if necessary delete old images in it
    import os
    if not os.path.exists(path_to_save_debug_convoluted_images_to):
        os.makedirs(path_to_save_debug_convoluted_images_to)
    # else:  # folder already exists -> delete it and recreate
    #     import shutil
    #     shutil.rmtree(path_to_save_debug_convoluted_images_to)
    #     os.makedirs(path_to_save_debug_convoluted_images_to)
    import Data as Data_manager
    for i in range(0, resulted_image.shape[1]):
        image = resulted_image[0, i, :, :]
        imagename = "result_layer"+str(layer_number)+"_filter"+str(i)+".png"
        Data_manager.Data.save_image(path_to_save_debug_convoluted_images_to + "/" + imagename, image)
