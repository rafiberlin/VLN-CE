from scripts.read_trajectories import read_compressed_json_file, \
    filter_preprocessed_data, \
    save_compressed_json_file
import os
import shutil
import argparse
from string import Template

"""
Script can be used to create smaller split files for debugging purpose.
Create this directory first:
data/datasets/R2R_VLNCE_v1-3_preprocessed/debug
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reduce the train file annotation with a list of episodes.")
    parser.add_argument(
        "--split",
        default="train",
        metavar="",
        help="Name of the splits: train, val_seen, val_unseen",
        type=str,
    )

    parser.add_argument(
        "--episodes",
        default="4991",
        metavar="",
        help="Comma separated episode ids",
        type=str,
    )

    args = parser.parse_args()
    episode_filter = [int(e) for e in args.episodes.strip().split(",")]

    split = args.split

    directory = '../data/datasets/R2R_VLNCE_v1-3_preprocessed/$split/'
    split_template = Template(directory + "$split.json.gz")
    split_template_gt = Template(directory + '$split')
    # work around, as the template sees the underscore as Regex character
    suffix = "_gt.json.gz"

    debug_split = "debug"

    outpath = Template(directory).substitute(split=debug_split)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    train_file = read_compressed_json_file(split_template.substitute(split=split))
    # That was a problematic episode when creating a video for the gold path

    print("Filtering:", episode_filter)
    filter_preprocessed_data(train_file, episode_filter)
    save_compressed_json_file(train_file, split_template.substitute(split=debug_split))
    outfile_gt = split_template_gt.substitute(split=debug_split) + suffix
    source_gt = split_template_gt.substitute(split=split) + suffix

    if os.path.exists(outfile_gt):
        os.remove(outfile_gt)
    shutil.copyfile(source_gt, outfile_gt)
    print("Done!")
