# import matplotlib.pyplot as plt

# from src.modules.color.ColorPallet import ColorPallet
# from src.modules.train.types import EpochResult


# class Visualizer:
#     @classmethod
#     def save_scatter(self, path: str, epoch_results: list[EpochResult]):
#         label = result["label"]
#         output = result["output"]

#         xy_min = min(label + output)
#         xy_max = max(label + output)

#         prop_name = result["prop_name"]

#         plt.figure(dpi=100, figsize=(8, 6))
#         plt.scatter(
#             label,
#             output,
#             color=ColorPallet.hex_universal_color["red"],
#             s=2,
#         )
#         plt.plot([xy_min, xy_max], [xy_min, xy_max], color=ColorPallet.hex_universal_color["blue"])
#         plt.xlabel(f"Observed {prop_name} value")
#         plt.ylabel(f"Predicted {prop_name} value")
#         plt.savefig(path)
#         plt.close()
