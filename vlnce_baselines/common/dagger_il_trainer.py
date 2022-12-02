from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
import lmdb


class DaggerILTrainer(BaseVLNCETrainer):
    """
    Ensures that a Dagger Based Trainer has an update_dataset method.
    Gives the possibility to create the dataset before starting a training,
    when we don t want to use Dagger
    """
    def __init__(self, config):
        super().__init__(config)

    def _update_dataset(self, data_it):
        raise NotImplementedError

    def create_dataset(self) -> None:
        """
        Method to simply generate the dataset before starting a training.
        You can first start the dataset creation with
        python run.py --exp-config vlnce_baselines/config/r2r_baselines/decision_transformer.yaml --run-type create_dataset
        It is practical if you do not want to use Data Agreggation for training, the dataset does not need to be generated each time.
        DAGGER:
            iterations: 1
            preload_lmdb_features: True
        :return:
        """
        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.config.IL.DAGGER.lmdb_map_size),
        ) as lmdb_env, lmdb_env.begin(write=True) as txn:
            txn.drop(lmdb_env.open_db())

        EPS = self.config.IL.DAGGER.expert_policy_sensor
        if EPS not in self.config.TASK_CONFIG.TASK.SENSORS:
            self.config.TASK_CONFIG.TASK.SENSORS.append(EPS)

        self.config.defrost()

        # if doing teacher forcing, don't switch the scene until it is complete
        if self.config.IL.DAGGER.p == 1.0:
            self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
                -1
            )
        self.config.freeze()

        observation_space, action_space = self._get_spaces(self.config)

        self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        print("Starting Dataset...")
        self._update_dataset(0 + (1 if self.config.IL.load_from_ckpt else 0))
        print("Dataset creation completed!")
