from os.path import dirname, abspath
from torch import cuda
from src.DataHelpers import DataManager
from src.ModelHelpers import ModelLoader
from src.Models import SentenceClassifier
from src.Predictor import Predictor


def prepare_data(data):
    return zip(*[((x.tokenids_pad, x.pad_mask), x.label) for x in data])


# Global vars.
CURRENT_DIR = dirname(abspath(__file__))
DEVICE = 'cuda' if cuda.is_available() else 'cpu'
MODEL_VERSION = "model_0"

# Load data.
data_manager = DataManager(process_data=False)
data_valid, _ = data_manager.load_processed_valid_data()
sents, true_labels = prepare_data(data_valid)

# Load model.
classifier = SentenceClassifier()
model_loader = ModelLoader(
    model=classifier,
    saves_folder=f"{CURRENT_DIR}/models_saves/{MODEL_VERSION}"
    )
classifier, vocabulary, label2id = model_loader.load()
classifier.to(DEVICE)

predictor = Predictor(classifier, DEVICE)
pred_labels = predictor.predict(sents)

count = 0
for datum, pred_label, true_label in zip(data_valid, pred_labels, true_labels):
    if pred_label != true_label:
        print(datum.text)
        print("true : ", true_label)
        print("pred : ", pred_label)
        print("\n")
        count += 1

print("\n\nAll sentences : ", len(data_valid))
print("Mistakes : ", count)
