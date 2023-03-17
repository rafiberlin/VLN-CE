import os
from collections import defaultdict, OrderedDict
import json
import pandas as pd
import shutil


def get_result_files_per_datasplit(eval_dir):
    list_result = None
    if os.path.isdir(eval_dir):
        print("Running several config files from:", eval_dir)
        list_result = defaultdict(OrderedDict)
        for file in os.listdir(eval_dir):
            if file.endswith(".json") and "predictions" not in file:
                file_path = os.path.join(eval_dir, file)
                # The file name is like this: stats_ckpt.4.pth_val_seen.json
                # after this spliting, we get 4 and val_seen.json
                iteration, data_split = file_path.split("stats_ckpt.")[1].split(".pth_")
                iteration = int(iteration)
                # we get val_seen
                data_split = data_split.split(".json")[0]
                list_result[data_split][iteration] = file_path
                # print("exp_config", file_path)

    return list_result


def read_results_per_split(result_path_dict, split=None):
    list_of_poor_iterations = {}
    for data_split in result_path_dict.keys():
        if split is not None and data_split != split:
            continue
        values = {}
        sorted_keys = sorted(result_path_dict[data_split].keys())
        for iteration in sorted_keys:
            with open(result_path_dict[data_split][iteration], 'r') as f:
                datapoint = json.load(f)
                values[iteration] = datapoint
        frame = pd.DataFrame.from_dict(values, columns=list(datapoint.keys()), orient="index")
        poor_iterations = [k for k in sorted_keys]
        list_of_poor_iterations[data_split] = poor_iterations, frame
    return list_of_poor_iterations


def read_poor_results_per_split(result_path_dict, keep_n_best=5, split=None, criteria="spl"):
    list_of_poor_iterations = read_results_per_split(result_path_dict, split)
    for data_split in list_of_poor_iterations.keys():
        if split is not None and data_split != split:
            continue
        _, frame = list_of_poor_iterations[data_split]
        bests = frame[criteria].nlargest(keep_n_best)
        poor_iterations = [k for k in result_path_dict[data_split].keys() if k not in bests.keys()]
        list_of_poor_iterations[data_split] = poor_iterations
    return list_of_poor_iterations


def move_poor_checkpoints(checkpoints_dir, poor_iterations):
    if os.path.isdir(checkpoints_dir):
        bad_dir = os.path.join(checkpoints_dir, "bad")
        if not os.path.exists(bad_dir):
            print(f"Bad iterations will be moved to: {bad_dir}")
            os.makedirs(bad_dir)
        for data_split in poor_iterations.keys():
            for iteration in poor_iterations[data_split]:
                file_name = f"ckpt.{str(iteration)}.pth"
                file_path = os.path.join(checkpoints_dir, file_name)
                if os.path.exists(file_path):
                    print(f"Moving: {file_name}")
                    shutil.move(file_path, os.path.join(bad_dir, file_name))


def list_best_result(result_dir, split, criteria, transformer_type=["normal", "enhanced", "full"], eval_dir="evals"):
    res_dict = {}
    keep_n_best = 1
    for upper_dir in transformer_type:
        path = os.path.join(result_dir, upper_dir)
        if os.path.exists(path):
            l = [d for d in os.listdir(path) if not d.startswith(".") and not d.startswith("_")]
            if len(l) > 0:
                for model in l:
                    model_result_dir = os.path.join(path, model, eval_dir)
                    if os.path.exists(model_result_dir):
                        result_files = get_result_files_per_datasplit(model_result_dir)
                        result_table = read_results_per_split(result_files, split=split)
                        res_dict[model_result_dir] = result_table

    best_score = 0.0
    best_model = "No model found"
    all_best = {}
    for model_result_dir in res_dict.keys():
        if split in res_dict[model_result_dir].keys():
            _, frame = res_dict[model_result_dir][split]
            best_index = frame[criteria].nlargest(keep_n_best)
            current_res = frame[criteria][best_index.index]
            metrics = current_res.values[0]
            all_best[model_result_dir] = metrics, best_index.index.values[0], frame.iloc[best_index]
            if metrics >= best_score:
                best_score = metrics
                best_model = model_result_dir

    print("Best:", best_model, split, best_score)
    return dict(sorted(all_best.items(), key=lambda item: item[1][0]))


if __name__ == "__main__" :
    transformer_type = ["enhanced"]  # ["normal", "enhanced", "full"]
    split = "val_seen"
    result_dir = "../data/checkpoints"
    criteria = "success"

    all_best_res = list_best_result(result_dir, split, criteria, transformer_type)

    print(f"\n################  {len(all_best_res)}  results retrieved for {split} #################\n")

    for k, v in all_best_res.items():
        print(k.split("data/")[1].split("/evals")[0], ",epoch:", v[1], f",best {criteria}:", v[0], ", other: ",
              {k: v[2][k].values[0] for k in v[2].keys() if k not in [criteria, "oracle_success", "path_length"]})


