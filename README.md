# Vision-and-Language Navigation in Continuous Environments (VLN-CE) with Transformer Models for R2R

[Project Website](https://jacobkrantz.github.io/vlnce/) — [VLN-CE Challenge](https://eval.ai/web/challenges/challenge-page/719) — [RxR-Habitat Challenge](https://ai.google.com/research/rxr/habitat)

Official implementations:

- *Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments* ([paper](https://arxiv.org/abs/2004.02857))
- *Waypoint Models for Instruction-guided Navigation in Continuous Environments* ([paper](https://arxiv.org/abs/2110.02207), [README](/vlnce_baselines/config/r2r_waypoint/README.md))

Vision and Language Navigation in Continuous Environments (VLN-CE) is an instruction-guided navigation task with crowdsourced instructions, realistic environments, and unconstrained agent navigation. This repo is a launching point for interacting with the VLN-CE task and provides both baseline agents and training methods. Both the Room-to-Room (**R2R**) and the Room-Across-Room (**RxR**) datasets are supported. VLN-CE is implemented using the Habitat platform.

<p align="center">
  <img width="775" height="360" src="./data/res/VLN_comparison.gif" alt="VLN-CE comparison to VLN">
</p>

## Hardware used

To give you an idea of the hardware and drivers at play:

We trained the models on A100 GPUs with nvidia drivers 510.85.02, with cuda 11.6 (headless).
For dev purposes, we used a RTX 3060 with nvidia drivers 470.161.03  and cuda 11.4 (with attached display, sufficient for transformers with 3 layers, 8 heads  and dimensions of 128).


## Setup

This project is developed with Python 3.6. If you are using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/), you can create an environment:


```bash
conda create -n vlnce python=3.6
conda activate vlnce
```

VLN-CE uses [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7 which can be [built from source](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation) or installed from conda:

```bash
# For installation on Remote servers / Virtual Machines without display output
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
# Or for machines with display output
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7
```



Then install a slightly a corrected version of [Habitat-Lab](https://github.com/rafiberlin/habitat-lab/tree/vlnce-fix):

The [original version]((https://github.com/facebookresearch/habitat-lab/tree/v0.1.7)) with commit "d6ed1c0a0e786f16f261de2beafe347f4186d0d8"
did not work with VLN-CE (due to some bug in the used version of OpenAI Gym). Further more, there is
a slight enhancement of the handling of evaluation mode (not waiting endlessly for new checkpoints to evaluate).

```bash
git clone --branch vlnce-fix git@github.com:rafiberlin/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
# that can takes a long time to build open-cv per wheel. Make sure update pip and have a gcc installed
# on you machine!
python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

At this point, you need to remove the installe version of torch that was automatically installed previously:

```bash
pip uninstall torch
```

And install this version along with torch vision:
```bash
# it worked even though the cuda version of the drivers was higher.
# But in case of problems, try to find the version working with your nvidia drivers
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Then, finish with:

```bash
git clone git@github.com:jacobkrantz/VLN-CE.git
cd VLN-CE
python -m pip install -r requirements.txt
```

Remark:

If you encounter error messages running the code, such as:

`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

Try to run the following (found in [Stackoverflow](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo)):

```bash
sudo apt install libgl1-mesa-glx libgl1
```

If you encounter error messages running the code, such as:

`ImportError: libEGL.so.1: cannot open shared object file: No such file or directory`

Try to run the following:

```bash
sudo apt install libegl1-mesa libegl1
```

If you encounter error messages running the code, such as:

`ImportError: libOpenGL.so.0: cannot open shared object file: No such file or directory`

Try to run the following:

```bash
sudo apt install libopengl0
```

### Data

#### Scenes: Matterport3D

Matterport3D (MP3D) scene reconstructions are used. The official Matterport3D download script (`download_mp.py`) can be accessed by following the instructions on their [project webpage](https://niessner.github.io/Matterport/). The scene data can then be downloaded:

```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

Extract such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`. There should be 90 scenes.

#### Episodes: Room-to-Room (R2R)

The R2R_VLNCE dataset is a port of the Room-to-Room (R2R) dataset created by [Anderson et al](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf) for use with the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator) (MP3D-Sim). For details on porting to 3D reconstructions, please see our [paper](https://arxiv.org/abs/2004.02857). `R2R_VLNCE_v1-3` is a minimal version of the dataset and `R2R_VLNCE_v1-3_preprocessed` runs baseline models out of the box. See the [dataset page](https://jacobkrantz.github.io/vlnce/data) for format, contents, and a changelog. We encourage use of the most recent version (`v1-3`).

| Dataset | Extract path | Size |
|-------------- |---------------------------- |------- |
| [R2R_VLNCE_v1-3.zip](https://drive.google.com/file/d/1qrdomxA5fuQ6n44NXzPAJe2dMdxatvma/view) | `data/datasets/R2R_VLNCE_v1-3` | 3 MB |
| [R2R_VLNCE_v1-3_preprocessed.zip](https://drive.google.com/file/d/1kQ_at68wiK2vAmlWbjJ4EDrLtaM0nfkR/view) | `data/datasets/R2R_VLNCE_v1-3_preprocessed` | 250 MB |

Downloading via CLI (only possible for ):

```bash
# R2R_VLNCE_v1-3
gdown https://drive.google.com/uc?id=1qrdomxA5fuQ6n44NXzPAJe2dMdxatvma
# R2R_VLNCE_v1-3_preprocessed
gdown https://drive.google.com/uc?id=1kQ_at68wiK2vAmlWbjJ4EDrLtaM0nfkR

```


Additionally, we provide some splits where data are separated by episode length.

Splits named *_50 contains all episodes from the original split up to an episode length of 50 steps.
Splits named *_50_plus contains all episodes from the original split with episode length higher than 50 steps.
Splits named *_50_ep or *_80_ep only contains respectively 50 and 80 episodes to speed up some evaluations.
(The script to create these split is under ./scripts/create_debug_split.py)


| Dataset | Extract path | Size |
|-------------- |---------------------------- |------- |
| [R2R_VLNCE_v1-3_preprocessed_additional_splits.zip](https://drive.google.com/file/d/1XqOd8t2KSdMvtJWSUziNGlM0ORZoYoUv/view) | `data/datasets/R2R_VLNCE_v1-3_preprocessed` | 265 MB |


Remark: Under:
data/datasets/R2R_VLNCE_v1-3_preprocessed/joint_train_envdrop/
You will need to rename the file  joint_train_envdrop.gz to joint_train_envdrop.json.gz


##### Encoder Weights

Baseline models encode depth observations using a ResNet pre-trained on PointGoal navigation. Those weights can be downloaded from [here](https://github.com/facebookresearch/habitat-lab/tree/master/habitat_baselines/rl/ddppo) (672M). Extract the contents to `data/ddppo-models/{model}.pth`.

#### Episodes: Room-Across-Room (RxR)

Download: [RxR_VLNCE_v0.zip](https://storage.googleapis.com/rxr-habitat/RxR_VLNCE_v0.zip)

About the [Room-Across-Room dataset](https://ai.google.com/research/rxr/) (RxR):

- multilingual instructions (English, Hindi, Telugu)
- an order of magnitude larger than existing datasets
- varied paths to break a shortest-path-to-goal assumption

RxR was ported to continuous environments originally for the [RxR-Habitat Challenge](https://ai.google.com/research/rxr/habitat). The dataset has `train`, `val_seen`, `val_unseen`, and `test_challenge` splits with both Guide and Follower trajectories ported. The starter code expects files in this structure:

```graphql
data/datasets
├─ RxR_VLNCE_v0
|   ├─ train
|   |    ├─ train_guide.json.gz
|   |    ├─ train_guide_gt.json.gz
|   |    ├─ train_follower.json.gz
|   |    ├─ train_follower_gt.json.gz
|   ├─ val_seen
|   |    ├─ val_seen_guide.json.gz
|   |    ├─ val_seen_guide_gt.json.gz
|   |    ├─ val_seen_follower.json.gz
|   |    ├─ val_seen_follower_gt.json.gz
|   ├─ val_unseen
|   |    ├─ val_unseen_guide.json.gz
|   |    ├─ val_unseen_guide_gt.json.gz
|   |    ├─ val_unseen_follower.json.gz
|   |    ├─ val_unseen_follower_gt.json.gz
|   ├─ test_challenge
|   |    ├─ test_challenge_guide.json.gz
|   ├─ text_features
|   |    ├─ ...
```

The baseline models for RxR-Habitat use precomputed BERT instruction features which can be downloaded from [here](https://github.com/google-research-datasets/RxR#downloading-bert-text-features) and saved to `data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{instruction_id}_{language}_text_features.npz`.

## RxR-Habitat Challenge

<p align="center">
  <img width="573" height="360" src="/data/res/rxr_teaser.gif" alt="RxR Challenge Teaser GIF">
</p>

**NEW: The 2022 RxR-Habitat Challenge is live!**

- Challenge webpage: [ai.google.com/research/rxr/habitat](https://ai.google.com/research/rxr/habitat)
- Workshop webpage: [embodied-ai.org](https://embodied-ai.org/)

The RxR-Habitat is hosted at the CVPR 2022 [Embodied AI workshop](https://embodied-ai.org/) set for June 19th, 2022. The leaderboard opens for challenge submissions on March 1. For official guidelines, please visit: [ai.google.com/research/rxr/habitat](https://ai.google.com/research/rxr/habitat). We encourage submissions on this dificult task!

The RxR-Habitat Challenge is hosted by Oregon State University, Google Research, and Facebook AI Research. This is the second year of the RxR-Habitat Challenge which first appeared at the 2021 CVPR [EAI workshop](https://embodied-ai.org/cvpr2021).

### Timeline

|               Event               |       Date      |
|:---------------------------------:|:---------------:|
|          Challenge Launch         |   Feb 14, 2022  |
|          Leaderboard Open         |   Mar 1, 2022  |
|         Leaderboard Closes        |   May 31, 2022  |
| Workshop and Winners Announcement | Jun 19, 2022 |

### Generating Submissions

Submissions are made by running an agent locally and submitting a jsonlines file (`.jsonl`) containing the agent's trajectories. Starter code for generating this file is provided in the function `BaseVLNCETrainer.inference()`. Here is an example of generating predictions for English using the Cross-Modal Attention baseline:

```bash
python run.py \
  --exp-config vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml \
  --run-type inference
```

If you use different models for different languages, you can merge their predictions with `scripts/merge_inference_predictions.py`. Submissions are only accepted that contain all episodes from all three languages in the `test-challenge` split. Starter code for this challenge was originally hosted in the `rxr-habitat-challenge` branch but is now integrated in `master`.

#### Required Task Configurations

As specified in the [challenge webpage](https://ai.google.com/research/rxr/habitat), submissions to the official challenge must have an action space of 30 degree turn angles, a 0.25m step size, and look up / look down actions of 30 degrees. The agent is given a 480x640 RGBD observation space. An example task configuration is given [here](/habitat_extensions/config/rxr_vlnce_english_task.yaml) which loads the English portion of the dataset.

The CMA baseline model ([config](/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml)) is an example of a valid submission. Existing [waypoint models](/vlnce_baselines/config/r2r_waypoint) are not valid due to their panoramic observation space. Such models would need to be adapted to the challenge configuration.

### Baseline Model

The official baseline for the RxR-Habitat Challenge is a monolingual cross-modal attention (CMA) model, labeled `Monolingual CMA Baseline` on the leaderboard. Configuration files for re-training or evaluating this model can be found in [this folder](vlnce_baselines/config/rxr_baselines) under the name `rxr_cma_{en|hi|te}.yaml`. Weights for the pre-trained models: [[en](https://drive.google.com/file/d/1wllkdat_wJZcFY0AyBoRlajiE2as2m6O/view?usp=sharing) [hi](https://drive.google.com/file/d/1GN6WbggQDTsYPsheRQFxE21Dr4UcIcab/view?usp=sharing) [te](https://drive.google.com/file/d/1B0v2QJL8gMUr-DlwAb-xVPzdN4coa7jD/view?usp=sharing)] (196MB each).

### Citing RxR-Habitat Challenge

To cite the challenge, please cite the following papers ([RxR](https://arxiv.org/abs/2010.07954) and [VLN-CE](https://arxiv.org/abs/2004.02857)):

```tex
@inproceedings{ku2020room,
  title={Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding},
  author={Ku, Alexander and Anderson, Peter and Patel, Roma and Ie, Eugene and Baldridge, Jason},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4392--4412},
  year={2020}
}

@inproceedings{krantz_vlnce_2020,
  title={Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments},
  author={Jacob Krantz and Erik Wijmans and Arjun Majundar and Dhruv Batra and Stefan Lee},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
 }
```

## Questions?

Feel free to contact the challenge organizers with any questions, comments, or concerns. The corresponding organizer is Jacob Krantz (@jacobkrantz). You can also open an issue with `[RxR-Habitat]` in the title, which will also notify us.

## VLN-CE Challenge (R2R Data)

The [VLN-CE Challenge](https://eval.ai/web/challenges/challenge-page/719) is live and taking submissions for public test set evaluation. This challenge uses the R2R data ported in the original VLN-CE paper.

To submit to the leaderboard, you must run your agent locally and submit a JSON file containing the generated agent trajectories. Starter code for generating this JSON file is provided in the function `BaseVLNCETrainer.inference()`. Here is an example of generating this file using the pretrained Cross-Modal Attention baseline:

```bash
python run.py \
  --exp-config vlnce_baselines/config/r2r_baselines/test_set_inference.yaml \
  --run-type inference
```

Predictions must be in a specific format. Please visit the challenge webpage for guidelines.

### Baseline Performance

The baseline model for the VLN-CE task is the cross-modal attention model trained with progress monitoring, DAgger, and augmented data (CMA_PM_DA_Aug). As evaluated on the leaderboard, this model achieves:

| Split      | TL   | NE   | OS   | SR   | SPL  |
|:----------:|:----:|:----:|:----:|:----:|:----:|
| Test       | 8.85 | 7.91 | 0.36 | 0.28 | 0.25 |
| Val Unseen | 8.27 | 7.60 | 0.36 | 0.29 | 0.27 |
| Val Seen   | 9.06 | 7.21 | 0.44 | 0.34 | 0.32 |

This model was originally presented with a val_unseen performance of 0.30 SPL, however the leaderboard evaluates this same model at 0.27 SPL. The model was trained and evaluated on a hardware + Habitat build that gave slightly different results, as is the case for the other paper experiments. Going forward, the leaderboard contains the performance metrics that should be used for official comparison. In our tests, the installation procedure for this repo gives nearly identical evaluation to the leaderboard, but we recognize that compute hardware along with the version and build of Habitat are factors to reproducibility.

For push-button replication of all VLN-CE experiments, see [here](vlnce_baselines/config/r2r_baselines/README.md).

## Starter Code

The `run.py` script controls training and evaluation for all models and datasets:

```bash
python run.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
```

For example, a random agent can be evaluated on 10 val-seen episodes of R2R using this command:

```bash
python run.py --exp-config vlnce_baselines/config/r2r_baselines/nonlearning.yaml --run-type eval
```

For lists of modifiable configuration options, see the default [task config](habitat_extensions/config/default.py) and [experiment config](vlnce_baselines/config/default.py) files.

Additionally to the original set of commands, we added the following:

```bash
python run.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {create_dataset | train_eval | check_dataset | train_complete}
```
create_dataset allow to create the dataset once. You won't need to recreate a new dataset each time a training is started
(works only for the Transformer related experiments.)
check_dataset only runs a basic to spot obvious errors in the dataset creation.
train_eval starts training and run the evaluation on one split.
train_complete starts training, evaluation on smaller validation set, on val seen, val unseen and finally creates the
prediction file for test.

Check also the template `vlnce_baselines/config/r2r_baselines/decision_transformer/templates/local/template.yaml`
to understand all the parameters needed for transformer models.

Furthermore you can run several config files in one command:

```bash
python run.py \
  --exp-config path/to/config_directory/ \
  --run-type {train | eval | inference | create_dataset | train_eval | check_dataset | train_complete}
```

### Training Agents

The `DaggerTrainer` class is the standard trainer and supports teacher forcing or dataset aggregation (DAgger). This trainer saves trajectories consisting of RGB, depth, ground-truth actions, and instructions to disk to avoid time spent in simulation.

The `RecollectTrainer` class performs teacher forcing using the ground truth trajectories provided in the dataset rather than a shortest path expert. Also, this trainer does not save episodes to disk, instead opting to recollect them in simulation.

Both trainers inherit from `BaseVLNCETrainer`.

### Evaluating Agents

Evaluation on validation splits can be done by running `python run.py --exp-config path/to/experiment_config.yaml --run-type eval`. If `EVAL.EPISODE_COUNT == -1`, all episodes will be evaluated. If `EVAL_CKPT_PATH_DIR` is a directory, each checkpoint will be evaluated one at a time.


### Transformer-Based Agents for R2R

In the file, vlnce_baselines/models/decision_transformer_policy.py, you need to subclass `AbstractDecisionTransformerNet`
to add your own model. Moreover, you would to allow your model's creation by adding its name in `vlnce_baselines/config/default.py`
in the variable `_C.MODEL.DECISION_TRANSFORMER.allowed_models`.

The training is then done with the `DecisionTransformerTrainer` (`decision_transformer` in the config file).

Pretrained models with corresponding training file under :


[Decision Transformer Agent](https://drive.google.com/file/d/1-E1l5g7DM36m3HYx8b4b4CNBC8d-OS83/view?usp=sharing)

[Enhanced Decision Transformer Agent](https://drive.google.com/file/d/1b2hpkHpiZIc2CBsaLzZWCa7qurfKDDsu/view?usp=sharing)

[Full Decision Transformer Agent](https://drive.google.com/file/d/1rS2_yo9_z35zzaHW4CtByorZ-jpDpht_/view?usp=sharing)

RemarK: `scripts/create_debug_split.py can be used to separate existing splits into smaller version based on episode length.

### Cuda

Cuda will be used by default if it is available. We find that one GPU for the model and several GPUs for simulation is favorable.

```yaml
SIMULATOR_GPU_IDS: [0]  # list of GPU IDs to run simulations
TORCH_GPU_ID: 0  # GPU for pytorch-related code (the model)
NUM_ENVIRONMENTS: 1  # Each GPU runs NUM_ENVIRONMENTS environments
```

The simulator and torch code do not need to run on the same device. For faster training and evaluation, we recommend running with as many `NUM_ENVIRONMENTS` as will fit on your GPU while assuming 1 CPU core per env.

## License

The VLN-CE codebase is [MIT licensed](LICENSE). Trained models and task datasets are considered data derived from the mp3d scene dataset. Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

## Citing

If you use VLN-CE in your research, please cite the following [paper](https://arxiv.org/abs/2004.02857):

```tex
@inproceedings{krantz_vlnce_2020,
  title={Beyond the Nav-Graph: Vision and Language Navigation in Continuous Environments},
  author={Jacob Krantz and Erik Wijmans and Arjun Majundar and Dhruv Batra and Stefan Lee},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
 }
```

If you use the RxR-Habitat data, please additionally cite the following [paper](https://arxiv.org/abs/2010.07954):

```tex
@inproceedings{ku2020room,
  title={Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding},
  author={Ku, Alexander and Anderson, Peter and Patel, Roma and Ie, Eugene and Baldridge, Jason},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4392--4412},
  year={2020}
}
```
