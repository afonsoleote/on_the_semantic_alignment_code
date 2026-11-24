import torch
import numpy as np

class _ActivationHook:
    """
    Pickleable
    """
    def __init__(self, extractor, name, idxs=None):
        self.extractor = extractor
        self.name = name
        self.idxs = idxs

    def hook(self, model, input, output):
        # if to_extract is list, only detach
        act = output.detach()

        # if to_extract is dict, get the specific activations
        if self.idxs is not None:
            act = act[..., self.idxs]

        self.extractor.activations[self.name] = act

class ActivationsExtractor:
    def __init__(self, model, to_extract):
        """
        to_extract:
            str -> single layer
            list[str] -> multiple layers
            dict[layer_name -> neuron_indices]
        """
        self.model = model # the pytorch model
        self.to_extract = to_extract # the layers/neurons to extract
        self.activations = {} # dict to store activations during forward pass
        self.hooks = [] # keeps references to the hooks

        self._get_named_modules()
        self._set_hooks()


    def _get_named_modules(self):
        self.modules =  dict(self.model.named_modules())

    def _set_hooks(self):
        # convert str to list
        if isinstance(self.to_extract, str):
            self.to_extract = [self.to_extract]

        if isinstance(self.to_extract, list):
            for l in self.to_extract:
                layer = self.modules[l]
                self.hooks.append(layer.register_forward_hook(self._get_hook(l)))

        elif isinstance(self.to_extract, dict):
            for l, n in self.to_extract.items():
                layer = self.modules[l]
                self.hooks.append(layer.register_forward_hook(self._get_hook(l, n)))

    def _get_hook(self, name, idxs=None):
        return _ActivationHook(self, name, idxs).hook

    def extract(self, x, as_numpy=True, flatten=True):
        # reset stored activations
        self.activations = {}

        # forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x) # forward hooks will run here and be stored in self.activations

        if not flatten:
            if as_numpy:
                return {k: v.cpu().numpy() for k, v in self.activations.items()}
            return self.activations

        acts = torch.cat([v.reshape(v.shape[0], -1) for v in self.activations.values()], dim=1)

        return acts.cpu().numpy() if as_numpy else acts

    def close(self):
        """Close the existing hooks. Note: should always be called when an
         ActivationsExtractor is no longer going be used to avoid accumulating
         unnecessary hooks on a model."""
        for h in self.hooks:
            h.remove()

    def renew(self, model=None, to_extract=None):
        self.close()

        if model is not None:
            self.model = model

        if to_extract is not None:
            self.to_extract = to_extract

        self.activations = {} # dict to store activations during forward pass
        self.hooks = [] # keeps references to the hooks

        self._get_named_modules()
        self._set_hooks()

#testing

if __name__ == "__main__":
    from src.utils.common_paths import MAIN_NNS_PATH
    from src.utils.nn_utils.checkpoint_utils import load_checkpoint
    from src.utils.models.type_a_model_recover import ConvExp16Recover
    from torchvision import transforms
    from PIL import Image

    state_dict, config = load_checkpoint(MAIN_NNS_PATH / "TypeA.pth", device="cpu")

    model = ConvExp16Recover.build(3)

    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    image_paths = [
        "../../../other_repos/XTRAINS_generator/output/images/cargo_type_circle/000000091.png",
        "../../../other_repos/XTRAINS_generator/output/images/cargo_type_circle/000000091_cf.png",
    ]

    preprocess = transforms.Compose([
        transforms.Resize((152, 152)),
        transforms.ToTensor(),
    ])

    img_tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img)
        img_tensors.append(img_tensor)

    batch = torch.stack(img_tensors)
    batch = batch.to("cpu")

    ae = ActivationsExtractor(model, ["33", "36"])
    acts = ae.extract(batch, flatten=False)
    print(acts)

    ae = ActivationsExtractor(
        model,
        {
            "33": [0, 4, 10],
            "36": [2, 5]
        }
    )
    acts = ae.extract(batch, flatten=True)
    print(acts)
    exit()
    extractor = ActivationsExtractor(model, ["36"])
    extractor = ActivationsExtractor(model, ["33", "36"])


