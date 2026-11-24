from pathlib import Path
from pprint import pformat


class ExperimentConfig:
    def __init__(self, params, identity=None, path_preamble=None):
        self.params = params
        self.identifying = identity or list(self.params.keys())
        self.path_preamble = path_preamble

    def __param_str(self, v):
        val = self[v]
        if callable(val):
            return val.__name__
        return str(val)

    def pseudo_path(self, param_keys=None):
        if param_keys is None:
            param_keys = self.identifying

        for key in param_keys:
            if isinstance(key, (list, tuple)):
                yield '_'.join([self.__param_str(k) for k in key])
            else:
                yield self.__param_str(key)

    def get_path(self):
        if self.path_preamble:
            path = tuple((self.path_preamble, *self.pseudo_path()))
        else:
            path = tuple(self.pseudo_path())

        return Path(*path)

    def has_run(self):
        exp_path = self.get_path()

        has_results = (exp_path / 'results.pickle').exists()
        has_config = (exp_path / 'config.pickle').exists()

        assert has_results == has_config, \
            f"Only partial results for {exp_path}!"

        return has_results

    # make access to parameters similar to a dictionary
    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __contains__(self, key):
        return key in self.params

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def update(self, other):
        for key, value in other.items():
            self.params[key] = value

    def __repr__(self):
        return str(self.get_path())

    def __str__(self):
        return pformat(self.params, indent=4, width=80)
