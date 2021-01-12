from os.path import dirname, abspath
from torch import cuda
from src.DataHelpers import DataManager
from src.Models import SentenceClassifier
from src.Trainer import Trainer
from src.ModelHelpers import ModelLoader


def prepare_data(data):
    return [(x.tokenids_pad, x.pad_mask, x.label) for x in data]


# Global vars.
CURRENT_DIR = dirname(abspath(__file__))
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MODEL_VERSION = "model_0"
EPOCHS = 30
LEARNING_RATE = 5e-4

# Load data.
data_manager = DataManager(process_data=False)
data_train, vocabulary = data_manager.load_processed_train_data()
data_valid, _ = data_manager.load_processed_valid_data()
sents_train = prepare_data(data_train)
sents_valid = prepare_data(data_valid)

# Stat classifier.
classifier = SentenceClassifier(
    vocab_size=len(vocabulary),
    hidden_size=50,
    seq_len=len(sents_train[0][0]),
    n_heads=2,
    n_classes=2,
    trans_depth=4
    )
classifier.init_weights()
classifier.to(DEVICE)

# Train classifier.
trainer = Trainer(
    model=classifier,
    data_train=sents_train,
    data_valid=sents_valid,
    learning_rate=LEARNING_RATE,
    device=DEVICE
    )
best_trained_classifier = trainer.train(n_epochs=EPOCHS, batch_size=50)

# Save model.
model_saver = ModelLoader(
    model=best_trained_classifier,
    saves_folder=f"{CURRENT_DIR}/models_saves/{MODEL_VERSION}",
    vocabulary=vocabulary,
    label2id={'rel': 1, 'norel': 0}
    )
model_saver.save()
