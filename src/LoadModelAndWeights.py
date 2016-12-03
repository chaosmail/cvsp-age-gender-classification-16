import json
from keras.models import model_from_json


# load model and corresponding weights.
# File path is defined in config file
from keras.optimizers import SGD


def load_weights(model):
    settings = json.load(open("config", "r"))
    path_to_saved_weights = settings["path_to_saved_weights"]
    model.load_weights(path_to_saved_weights)
    return model


def load_model():
    print("Loading trained model...")
    settings = json.load(open("config", "r"))
    path_to_saved_model = settings["path_to_saved_model"]
    learn_rate = float(settings["learn_rate"])
    model = model_from_json(open(path_to_saved_model, "r").read())
    sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    return model
