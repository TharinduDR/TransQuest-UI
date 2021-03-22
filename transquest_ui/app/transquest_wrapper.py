import logging
import os

from google_drive_downloader import GoogleDriveDownloader as gdd
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel


class PredictedToken:
    def __init__(self, text, quality):
        self.text = text
        self.quality = quality


class MicroTransQuestWrapper:
    def __init__(self, model_name_or_path, model_type=None, use_cuda=True, cuda_device=-1):

        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device

        MODEL_CONFIG = {
            "en_de": ("xlmroberta", "108feANZ_VTaAjJMwR85DPzzN6TLdLjvd"),
            "en_zh": ("xlmroberta", "108feANZ_VTaAjJMwR85DPzzN6TLdLjvd"),
        }

        if model_name_or_path in MODEL_CONFIG:
            self.trained_model_type, self.drive_id = MODEL_CONFIG[model_name_or_path]

            try:
                from torch.hub import _get_torch_home
                torch_cache_home = _get_torch_home()
            except ImportError:
                torch_cache_home = os.path.expanduser(
                    os.getenv('TORCH_HOME', os.path.join(
                        os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
            default_cache_path = os.path.join(torch_cache_home, 'transquest')
            self.model_path = os.path.join(default_cache_path, self.model_name_or_path)
            if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
                logging.info(
                    "Downloading TransQuest model and saving it at {}".format(self.model_path))

                gdd.download_file_from_google_drive(file_id=self.drive_id,
                                                    dest_path=os.path.join(self.model_path, "model.zip"),
                                                    showsize=True, unzip=True)

            model_args = {"use_multiprocessing": False}
            self.model = MicroTransQuestModel(self.trained_model_type, self.model_path, args=model_args,
                                              use_cuda=self.use_cuda,
                                              cuda_device=self.cuda_device)

        else:
            model_args = {"use_multiprocessing": False}
            self.model = MicroTransQuestModel(model_type, self.model_name_or_path, args=model_args,
                                              use_cuda=self.use_cuda,
                                              cuda_device=self.cuda_device)

    @staticmethod
    def _download(drive_id, model_name):
        gdd.download_file_from_google_drive(file_id=drive_id,
                                            dest_path=os.path.join(".transquest", model_name, "model.zip"),
                                            unzip=True)

    def predict_quality(self, source: str, target: str):

        source_words = source.split()
        target_words = target.split()

        logging.info(source_words)
        logging.info(target_words)

        source_tags, target_tags = self.model.predict([[source, target]])

        logging.info(source_tags)
        logging.info(target_tags)

        source_predicted_tokens = []
        target_predicted_tokens = []
        for source_word, source_tag in zip(source_words, source_tags[0]):
            source_predicted_token = PredictedToken(source_word, source_tag)
            source_predicted_tokens.append(source_predicted_token)

        gap_index = 0
        word_index = 0
        for prediction_id, prediction in enumerate(target_tags[0]):
            if prediction_id % 2 == 0:
                target_predicted_token = PredictedToken("<GAP>", prediction)
                gap_index += 1
                target_predicted_tokens.append(target_predicted_token)
            else:
                target_predicted_token = PredictedToken(target_words[word_index], prediction)
                word_index += 1
                target_predicted_tokens.append(target_predicted_token)

        return source_predicted_tokens, target_predicted_tokens


class MonoTransQuestWrapper:
    def __init__(self, model_name_or_path, model_type=None, use_cuda=True, cuda_device=-1):

        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.use_cuda = use_cuda
        self.cuda_device = cuda_device

        MODEL_CONFIG = {
            "en_de_hter": ("xlmroberta", "1YYL2qCtceHa1ufrLwF-iBlrYugAF2lE1"),
            "en_de_da": ("xlmroberta", "1byzONzC1t1Qc0m76c4TOEGvC0yMBWnMU"),
        }

        if model_name_or_path in MODEL_CONFIG:
            self.trained_model_type, self.drive_id = MODEL_CONFIG[model_name_or_path]

            try:
                from torch.hub import _get_torch_home
                torch_cache_home = _get_torch_home()
            except ImportError:
                torch_cache_home = os.path.expanduser(
                    os.getenv('TORCH_HOME', os.path.join(
                        os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
            default_cache_path = os.path.join(torch_cache_home, 'transquest')
            self.model_path = os.path.join(default_cache_path, self.model_name_or_path)
            if not os.path.exists(self.model_path) or not os.listdir(self.model_path):
                logging.info(
                    "Downloading TransQuest model and saving it at {}".format(self.model_path))

                gdd.download_file_from_google_drive(file_id=self.drive_id,
                                                    dest_path=os.path.join(self.model_path, "model.zip"),
                                                    showsize=True, unzip=True)

            model_args = {"use_multiprocessing": False}
            self.model = MonoTransQuestModel(self.trained_model_type, self.model_path, args=model_args,
                                             use_cuda=self.use_cuda,
                                             cuda_device=self.cuda_device)

        else:
            model_args = {"use_multiprocessing": False}
            self.model = MonoTransQuestModel(model_type, self.model_name_or_path, args=model_args,
                                             use_cuda=self.use_cuda,
                                             cuda_device=self.cuda_device)

    @staticmethod
    def _download(drive_id, model_name):
        gdd.download_file_from_google_drive(file_id=drive_id,
                                            dest_path=os.path.join(".transquest", model_name, "model.zip"),
                                            unzip=True)

    def predict_quality(self, source: str, target: str):
        predictions, raw_outputs = self.model.predict([[source, target]])
        return predictions




