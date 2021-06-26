import os
from torchaudio.datasets import SPEECHCOMMANDS


class Dataset(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
