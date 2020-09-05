import numpy as np

class SummaryRow():
    """
    Helper class to sumarize results for ndarrays
    """

    _HEADER = [
        "row_header",
        "shape",
        "min",
        "max",
        "mean",
        "std"
    ]

    _FORMAT = {
        "row_header": "{:25s}",
        "shape": "{}",
        "min": "{:+.4e}",
        "max": "{:+.4e}",
        "mean": "{:+.4e}",
        "std": "{:+.4e}"
    }

    def __init__(self, arr: np.ndarray, row_header: str = ""):
        self.row_header = row_header
        self.shape = arr.shape
        self.min = arr.min()
        self.max = arr.max()
        self.mean = arr.mean()
        self.std = arr.std()

    @staticmethod
    def print_header():
        header = "|".join(SummaryRow._HEADER)
        return header

    def __str__(self):
        values = [
            self._FORMAT["row_header"].format(self.row_header),
            self._FORMAT["shape"].format(self.shape),
            self._FORMAT["min"].format(self.min),
            self._FORMAT["max"].format(self.max),
            self._FORMAT["mean"].format(self.mean),
            self._FORMAT["std"].format(self.std)
        ]

        formated_values_repr = "|".join(values)
        return formated_values_repr