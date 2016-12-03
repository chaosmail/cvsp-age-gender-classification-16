from Evaluate import evaluate_manually, plot_model, evaluate_with_keras_method, evaluate_manually_with_training_data, show_layer
import LoadModelAndWeights
import Data as DataManager

print("Start evaluation...")

# Load saved (and trained) model
model = LoadModelAndWeights.load_model()
# plot_model(model)
LoadModelAndWeights.load_weights(model)
data = DataManager.Data()
# evaluate_manually(model, data)
evaluate_with_keras_method(model, data)
# test_image = data.train_images[0, :, :, :].reshape((1, 1, 48, 48))
# for i in range(0, 20):
#     show_layer(model, i, test_image)


evaluate_manually_with_training_data(model, data)
