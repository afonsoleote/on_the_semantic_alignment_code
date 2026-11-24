import os
import pickle
import shutil

class ExperimentModel:
    """Class representing a model from some experiment.
    This class preserves relevant context for the model and its experiment.
    """

    CKPT_FILE = 'model.pickle'
    STATE_FILE = 'obj_state.pickle'

    def __init__(self, model, model_class, data_ids, config):
        """
        Parameters
        ----------
        model: Any
            A model.
        model_class: type
            The model's Python class.
        data_ids: dict
            Dictionary of labels to data identifiers, stores what data was used
             for which purpose during the model's development.
        config: dict
            Experiment configuration used during the model's development.
        """
        # model parameters
        self.model = model
        self.model_class = model_class

        # data used while developing the model
        self.data_ids = data_ids

        # experiment configuration used to develop the model
        self.config = config

        # store other relevant information
        self.info = dict()

    def add_info(self, key, value):
        self.info[key] = value

    def get_info(self, key):
        return self.info[key]

    def _save_model(self, folder_path):
        with open(folder_path / ExperimentModel.CKPT_FILE, 'wb') as file:
            pickle.dump(self.model, file)

    def _save_object_state(self, folder_path):
        temp_model = self.model
        del self.model

        with open(folder_path / ExperimentModel.STATE_FILE, 'wb') as file:
            pickle.dump(self, file)

        self.model = temp_model

    def save(self, folder_path):
        # create folder to store data
        os.makedirs(folder_path, exist_ok=True)

        # save model
        self._save_model(folder_path)

        # save object state
        self._save_object_state(folder_path)

