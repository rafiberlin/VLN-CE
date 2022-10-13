#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
#Needed import to load VLNCE Envs
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from scripts.read_trajectories import read_compressed_json_file
from habitat_extensions.utils import observations_to_image, generate_video
from habitat.utils.visualizations.utils import append_text_to_image
cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def shortest_path_example():
    file2 = "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz"
    ground_truth_annotaions = read_compressed_json_file(file2)
    config = get_config("vlnce_baselines/config/r2r_baselines/seq2seq_da.yaml", None)
    config.defrost()
    if len(config.VIDEO_OPTION) > 0:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    config.freeze()

    from PIL import Image


    with construct_envs(config, get_env_class(config.ENV_NAME)) as env:


        print("Environment creation successful")
        for episode in range(3):
            observations = env.reset()
            episodes = env.current_episodes()
            #trajectory_ids = [ o['instruction']['trajectory_id'] for o in observations ]
            episode_ids = [ e.episode_id for e in episodes ]
            ground_truth_actions = [ ground_truth_annotaions[id] for id in episode_ids ]
            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)
            print("Agent stepping around inside environment.")
            images = []

            rgb_frames = [[] for _ in range(env.num_envs)]

            for i in range(env.num_envs) :
                print("Episode:", episode_ids[i])
                for action in ground_truth_actions[i]["actions"]:
                    observation, _, _, infos = env.step_at(i,action)
                    if len(config.VIDEO_OPTION) > 0:
                        frame = observations_to_image(observation, infos)
                        frame = append_text_to_image(
                            frame, observation["instruction"]["text"]
                        )
                        rgb_frames[i].append(frame)
                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=episode_ids[i],
                        checkpoint_idx=0,
                        metrics={"NONE":0.0},
                        tb_writer=None
                    )
                rgb_frames[i] = []
                exit(1)
            print("Episode finished")


def main():
    shortest_path_example()


if __name__ == "__main__":
    main()
