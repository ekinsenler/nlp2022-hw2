import json
import random
from typing import Dict
import numpy as np
from typing import List, Tuple
from dataset import PICDataset
from torch.utils.data import DataLoader
from model import Model
import torch
from stud_model import SRLModel
import pathlib


def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    ###return Baseline(language=language)
    return StudentModel(language=language, device=device)


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str, device: str):
        super(Model, self).__init__()
        # load the specific model for the input language
        self.language = language
        assert language == 'EN', 'Only english is implemented'
        self.device = torch.device(device)

        curr_dir = pathlib.Path(__file__)
        proj_dir = curr_dir.parent.parent.parent
        hw1_dir = curr_dir.parent.parent

        data_train_path = proj_dir / 'data' / 'data_hw2' / 'EN' / 'train.json'
        data_dev_path = proj_dir / 'data' / 'data_hw2' / 'EN' / 'dev.json'
        model_dir = proj_dir / 'model'

        if (model_dir / 'vocab.pt').is_file():
            self.vocab = torch.load(model_dir / 'vocab.pt')
        else:
            self.vocab = Vocabulary()
            self.vocab.construct_vocabulary(train_sentences, train_labels)
            torch.save(vocab, model_dir / 'vocab.pt')
        self.SRLModel = SRLModel(vocab=self.vocab, device=self.device)
        #self.SRLModel.to(self.device)

    def predict_sentences(self, sentences_dataloader:DataLoader):
        result = dict()
        for id in sentences_dataloader.dataset.no_pred_ids:
            result[id] = {'roles': {}}
        self.SRLModel.eval()
        with torch.no_grad():
            for batch_sentence in sentences_dataloader:
                batch_prediction = self.SRLModel(batch_sentence)
                batch_padding_masking = batch_sentence['words'] > 0
                for prediction, mask, gold_pred, key in zip(batch_prediction, batch_padding_masking, batch_sentence['predicates'], batch_sentence['id']):
                    prediction_index = torch.argmax(prediction, -1)
                    #unpadded_pred = torch.masked_select(prediction, mask)
                    unpadded_pred = dict()
                    unpadded_pred['predicates'] = torch.masked_select(gold_pred, mask)
                    unpadded_pred['predicates'] = self.vocab.indices2preds(unpadded_pred['predicates'])
                    unpadded_pred['pred_index'] = sentences_dataloader.dataset.get_predicates(unpadded_pred)
                    prediction_str = self.vocab.indices2roles(prediction_index)
                    if unpadded_pred['pred_index']:
                        for i in unpadded_pred['pred_index'].keys():
                            if key in result:
                                result[key]['roles'][i] = prediction_str
                            else:
                                result[key] = {'roles': {i: prediction_str}}
                    else:
                        result[key] = {'roles': {}}
        return result

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence.
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                    }
        """
        test_dataset = PICDataset({0 : sentence}, labels=None, vocab=self.vocab)
        test_dataset.prepare_sentences()
        result = self.predict_sentences(test_dataset)

        return result[0]
