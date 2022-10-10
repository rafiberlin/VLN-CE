from scripts.read_trajectories import read_compressed_json_file,\
    filter_preprocessed_data,\
    save_compressed_json_file

"""
Script can be used to create smaller split files for debugging purpose.
Create this directory first:
data/datasets/R2R_VLNCE_v1-3_preprocessed/debug
"""
if __name__ == "__main__":
    file1 = "../data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
    outpath = "../data/datasets/R2R_VLNCE_v1-3_preprocessed/debug/debug.json.gz"
    train_file = read_compressed_json_file(file1)
    # That was a problematic episode when creating a video for the gold path
    episode_filter = [118]
    print("Filtering:", episode_filter)
    filter_preprocessed_data(train_file, episode_filter)
    save_compressed_json_file(train_file, outpath)
    print("Done!")
