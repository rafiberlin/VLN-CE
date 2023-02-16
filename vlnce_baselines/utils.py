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
            if file.endswith(".json"):
                file_path = os.path.join(eval_dir, file)
                # The file name is like this: stats_ckpt.4.pth_val_seen.json
                # after this spliting, we get 4 and val_seen.json
                iteration, data_split = file_path.split("stats_ckpt.")[1].split(".pth_")
                iteration = int(iteration)
                #we get val_seen
                data_split = data_split.split(".json")[0]
                list_result[data_split][iteration] = file_path
                print("exp_config", file_path)

    return  list_result
def read_results_per_split(result_path_dict, keep_n_best = 5, split=None, criteria="spl"):
    list_of_poor_iterations = {}
    for data_split in result_path_dict.keys():
        if split is not None and data_split != split:
            continue
        values = {}
        for iteration in reversed(result_path_dict[data_split].keys()):
            with open(result_path_dict[data_split][iteration], 'r') as f:
                datapoint = json.load(f)
                values[iteration] = datapoint
        frame = pd.DataFrame.from_dict(values, columns=list(datapoint.keys()), orient="index")
        bests = frame[criteria].nlargest(keep_n_best)
        poor_iterations = [ k for k in result_path_dict[data_split].keys() if k not in bests.keys() ]
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

