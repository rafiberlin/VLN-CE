from scripts.read_trajectories import read_compressed_json_file, \
    filter_preprocessed_data, \
    save_compressed_json_file, \
    get_episode_list
import os
import shutil
import argparse
from string import Template
from pathlib import Path

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
        default="val_seen",
        metavar="",
        help="Name of the splits: train, val_seen, val_unseen, joint_train_envdrop",
        type=str,
    )

    parser.add_argument(
        "--episodes",
        default="773,1818,261,7645,90,91",
        metavar="",
        help="Comma separated episode ids",
        type=str,
    )
    parser.add_argument(
        "--lower-length",
        default=1,
        metavar="",
        help="Episode Length min",
        type=int,
    )

    parser.add_argument(
        "--upper-length",
        default=200,
        metavar="",
        help="Episode Length max",
        type=int,
    )

    args = parser.parse_args()
    episode_filter = [int(e) for e in args.episodes.strip().split(",")]
    boundaries = (args.lower_length, args.upper_length)
    split = args.split

    ep_limit = 50

    directory = '../data/datasets/R2R_VLNCE_v1-3_preprocessed/$split/'
    split_template = Template(directory + "$split.json.gz")
    split_template_gt = Template(directory + '$split')
    # work around, as the template sees the underscore as Regex character
    suffix = "_gt.json.gz"
    debug_split = "val_seen_50_ep"

    outfile_gt = split_template_gt.substitute(split=debug_split) + suffix
    source_gt = split_template_gt.substitute(split=split) + suffix
    train_file_gt = read_compressed_json_file(source_gt)

    outpath = Template(directory).substitute(split=debug_split)

    length_list = get_episode_list(train_file_gt, boundaries, ep_limit)
    if len(length_list) > 0:
        print("Using filter based on episode length")
        episode_filter = length_list
    if not os.path.exists(outpath):
        print("create", outpath)
        os.makedirs(outpath)
    train_file = read_compressed_json_file(split_template.substitute(split=split))


    filter_preprocessed_data(train_file, episode_filter)
    debug_outfile = split_template.substitute(split=debug_split)
    # if not os.path.exists(debug_outfile):
    #     print("Create:", debug_outfile)
    #     Path(debug_outfile).touch()
    save_compressed_json_file(train_file, debug_outfile)


    if os.path.exists(outfile_gt):
        os.remove(outfile_gt)
    shutil.copyfile(source_gt, outfile_gt)
    print("Done!")
