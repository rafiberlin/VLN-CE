import json
import gzip
import statistics


def read_compressed_json_file(path):
    """
    Reading a compressed json file
    :param path:
    :return:
    """

    with gzip.open(path, "rb") as f:
        file = json.loads(f.read().decode("utf-8"))
    return file


def output_vlnce_r2r_statistics():
    """
    Output some statistics
    :return:
    """

    splits = ["train", "val_seen", "val_unseen", "test"]

    for split in splits:
        print("#####################################################\n")
        print("Split: ", split)
        file = f"../data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
        annotation = read_compressed_json_file(file)
        episodes = annotation["episodes"]
        trajectories = set([e["trajectory_id"] for e in episodes])
        scenes = set([e["scene_id"] for e in episodes])
        vocab = annotation["instruction_vocab"]
        print(str(len(episodes)) + " episodes.")
        print(str(len(trajectories)) + " trajectories.")  # Each trajectory has 3 different navigation instructions
        print(str(len(scenes)) + " scenes.")
        print(str(vocab["num_vocab"]) + " words.")
        # 0 is use for padding, we count the number of tokens that are not 0
        print("Average instruction length: " + str(statistics.mean(
            [len(list(filter(lambda x: x != 0, e["instruction"]["instruction_tokens"]))) for e in episodes])))


def save_compressed_json_file(data, path):
    """
    Saving as a compressed json
    :param data: the data to be saved
    :param path: the complete path to save to
    :return:
    """

    print("Saving under: ", path)
    with gzip.open(path, 'w') as out_file:
        out_file.write(json.dumps(data).encode('utf-8'))


def filter_preprocessed_data(data, filter: list):
    """
    Opening one of the split files:
    "../data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
    can be used to only keep episodes listed in the filter
    :param data:
    :param filter:
    :return:
    """

    episodes = {data['episodes'][i]["episode_id"]: data["episodes"][i] for i in range(len(data["episodes"]))}
    data["episodes"] = [episodes[episode_id] for episode_id in filter]


if __name__ == "__main__":
    print("Reading...")
    file1 = "../data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
    file2 = "../data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
    train_file = read_compressed_json_file(file1)
    train_file_gt = read_compressed_json_file(file2)
    output_vlnce_r2r_statistics()
    print("Done!")
