import re
from src.globals import ENT_MASK


class SentObject:
    '''
    General format each data point should have.
    Includes all the fields that will be created from processing the original
    example.
    '''
    def __init__(self, sentence):
        self.text = ""
        self.label = 0
        self.annotation_type = ""
        self.source = ""
        self.text_tokens = []
        self.tokenids = []
        self.tokenids_pad = []
        self.pad_mask = []


class SentTrain(SentObject):
    '''
    Format examples from the train set.
    '''
    def __init__(self, example):
        super(SentTrain, self).__init__(example)
        self.text = example['sent']
        self.label = self.get_label(example['rel'])
        self.annotation_type = example['type']
        self.source = example['source']

    def get_label(self, rel):
        return 1 if rel != "norel" else 0


class SentValid(SentObject):
    '''
    Format examples from the validation set.
    '''
    def __init__(self, example):
        example = eval(example)
        super(SentValid, self).__init__(example)
        self.text = self.mask_entities(example)
        self.label = example[2]

    def mask_entities(self, example):
        subj = re.escape(example[0][0])
        obj = re.escape(example[0][1])
        text = example[1]
        text = re.sub(r"\b{}\b".format(subj), ENT_MASK, text)
        text = re.sub(r"\b{}\b".format(obj), ENT_MASK, text)
        return text
