import h5py
from tqdm import tqdm

from src.modules.train.types import EpochResult, PropEpochResult


def save_epoch_results_as_hdf5(path: str, epoch_results: list[EpochResult]):
    with h5py.File(name=path, mode="w") as f:
        f.create_group("result")
        print("result saving...")

        for epoch_result in tqdm(epoch_results):
            epoch_group = f.create_group(f'result/{epoch_result["epoch"]}')
            epoch_group_attrs = epoch_group.attrs
            epoch_group_attrs["epoch"] = epoch_result["epoch"]

            for result in epoch_result["results"]:
                prop_group = f.create_group(f'result/{epoch_result["epoch"]}/{result['prop_name']}')
                f.create_dataset(f'result/{epoch_result["epoch"]}/{result['prop_name']}/output', data=result["output"])
                f.create_dataset(f'result/{epoch_result["epoch"]}/{result['prop_name']}/label', data=result["label"])
                prop_group_attrs = prop_group.attrs
                prop_group_attrs["pearsonr"] = result["pearsonr"]
                prop_group_attrs["mean_squared_error"] = result["mean_squared_error"]
                prop_group_attrs["root_mean_squared_error"] = result["root_mean_squared_error"]
                prop_group_attrs["mean_absolute_error"] = result["mean_absolute_error"]


def epoch_results_from_hdf5(path: str):
    with h5py.File(path, mode="r") as f:
        epoch_results: list[EpochResult] = []

        epoch_keys = f["result"].keys()
        print("result loading...")
        for epoch_key in tqdm(epoch_keys):
            epoch_group = f[f"result/{epoch_key}"]
            epoch_group_attrs = epoch_group.attrs
            epoch = epoch_group_attrs["epoch"]

            prop_names = epoch_group.keys()
            prop_epoch_results: list[PropEpochResult] = []
            for prop_name in prop_names:
                prop_group = f[f"result/{epoch_key}/{prop_name}"]
                output = f[f"result/{epoch_key}/{prop_name}/output"]
                label = f[f"result/{epoch_key}/{prop_name}/label"]

                prop_group_attrs = prop_group.attrs
                pearsonr = prop_group_attrs["pearsonr"]
                mean_squared_error = prop_group_attrs["mean_squared_error"]
                root_mean_squared_error = prop_group_attrs["root_mean_squared_error"]
                mean_absolute_error = prop_group_attrs["mean_absolute_error"]

                prop_epoch_result: PropEpochResult = {
                    "prop_name": prop_name,
                    "label": label,
                    "output": output,
                    "pearsonr": pearsonr,
                    "mean_squared_error": mean_squared_error,
                    "root_mean_squared_error": root_mean_squared_error,
                    "mean_absolute_error": mean_absolute_error,
                }
                prop_epoch_results.append(prop_epoch_result)

            epoch_result: EpochResult = {"epoch": epoch, "results": prop_epoch_results}
            epoch_results.append(epoch_result)

    return epoch_results
