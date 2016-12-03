import LoadModelAndWeights


def classify(images):
    print("Start classification...")
    model = LoadModelAndWeights.load_model()
    LoadModelAndWeights.load_weights(model)
    classes = model.predict_classes(images, batch_size=32)
    return classes
