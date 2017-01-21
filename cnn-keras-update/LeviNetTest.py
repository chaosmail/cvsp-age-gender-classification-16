from models import *

params = {
    'name': 'CNN Age',
    'input_shape': (3, 112, 112),
    'n_classes': 10,
    'n_epochs': 100,
    'batchsize': 64,
    'dropout': 0.4,
    'momentum': 0.9,
    'learning_rate': 1e-2,
    'learning_rate_decay': 1e-3,
    'early_stopping_rounds': 20,
    'l2_reg': 2e-4,
    'use_class_weights': False
}
model = get_levinet(params)