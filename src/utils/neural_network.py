import lightning.pytorch as pl
import os
import pickle
import shutil

from src.utils.experiment_model import ExperimentModel
from src.utils.nn_utils import load_checkpoint

class NeuralNetwork(ExperimentModel):
    """Class representing a neural network.
    This class preserves relevant context for a neural network model.
    """

    CKPT_FILE = 'model.ckpt'

    def get_lightning_model(self):
        return self.model

    def get_pytorch_model(self):
        return self.model.get_pytorch_model()

    def save(self, folder_path, checkpoint):
        os.makedirs(folder_path, exist_ok=True)

        # save model
        if isinstance(checkpoint, pl.Trainer):
            checkpoint.save_checkpoint(folder_path / NeuralNetwork.CKPT_FILE)
        else:
            shutil.copyfile(checkpoint, folder_path / NeuralNetwork.CKPT_FILE)

        # save object state
        self._save_object_state(folder_path)

    @staticmethod
    def load(folder_path, device, load_kwargs=None):
        load_kwargs = load_kwargs or {}

        # load object state
        with open(folder_path / NeuralNetwork.STATE_FILE, 'rb') as file:
            neural_network = pickle.load(file)

        # load model
        neural_network.model = load_checkpoint(
            neural_network.model_class,
            folder_path / NeuralNetwork.CKPT_FILE,
            device,
            load_kwargs=load_kwargs)

        return neural_network
