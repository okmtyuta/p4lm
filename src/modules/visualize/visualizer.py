import os

import matplotlib.pyplot as plt

from src.modules.protein.protein_list import ProteinProp


class Visualizer:
    @classmethod
    def save_scatter(self, outputs: list[float], labels: list[float], prop_name: ProteinProp, path):
        xy_min = min(labels + outputs)
        xy_max = max(labels + outputs)

        plt.figure(dpi=100, figsize=(8, 6))
        plt.scatter(
            labels,
            outputs,
            color="#990099",
            s=2,
        )
        plt.plot([xy_min, xy_max], [xy_min, xy_max])
        plt.xlabel(f"Observed {prop_name}")
        plt.ylabel(f"Predicted {prop_name}")
        plt.savefig(path)
        plt.close()
