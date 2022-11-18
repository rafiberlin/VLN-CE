import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from vlnce_baselines.models.encoders.min_gpt import GPT
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.policy import ILPolicy


@BaselineRegistry.register_policy
class DecisionTransformerPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            DecisionTransformerNet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class DecisionTransformerNet(Net):
    """A baseline sequence to sequence network that performs single modality
    encoding of the instruction, RGB, and depth observations. These encodings
    are concatentated and fed to an RNN. Finally, a distribution over discrete
    actions (FWD, L, R, STOP) is produced.
    """


    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__()
        self.model_config = model_config

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
        )

        self.gpt_encoder = GPT(self.model_config.DECISION_TRANSFORMER)

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=False,
        )

        if model_config.SEQ2SEQ.use_prev_action:
            self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Init the RNN state decoder
        rnn_input_size = (
            self.instruction_encoder.output_size
            + model_config.DEPTH_ENCODER.output_size
            + model_config.RGB_ENCODER.output_size
        )

        if model_config.SEQ2SEQ.use_prev_action:
            rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        # size due to concatenation of instruction, depth, and rgb features
        input_state_size = self.instruction_encoder.output_size \
                           + model_config.DEPTH_ENCODER.output_size\
                            + model_config.RGB_ENCODER.output_size

        assert model_config.DECISION_TRANSFORMER.reward_type in ["POINT_GOAL_NAV_REWARD", "SPARSE_REWARD"]
        self.reward_type = model_config.DECISION_TRANSFORMER.reward_type

        self.embed_timestep = nn.Embedding(model_config.DECISION_TRANSFORMER.episode_horizon, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_return = nn.Linear(1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_state = nn.Linear(input_state_size, model_config.DECISION_TRANSFORMER.hidden_dim)
        #TODO: What do you want to use, linear or embedding? I guess it should be embedding...
        # But if you want linear you will have to modify your input entry.
        # instaed of having action 0, 1, 2 or 3 (5 if you try to add a start token), you will have a tensor of size (4,1) filled with 0 or 1.
        #  and for action 3, you will have a 1 in the 4th row and zero otherwise.
        self.embed_action = nn.Embedding(num_actions+1, model_config.DECISION_TRANSFORMER.hidden_dim)

        self.embed_ln = nn.LayerNorm(model_config.DECISION_TRANSFORMER.hidden_dim)

        self.progress_monitor = nn.Linear(
            self.model_config.STATE_ENCODER.hidden_size, 1
        )

        self.predict_action = nn.Sequential(
            *([nn.Linear(model_config.DECISION_TRANSFORMER.hidden_dim, num_actions+1)] + ([nn.Tanh()]))
        )

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self.model_config.STATE_ENCODER.hidden_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _init_layers(self):
        nn.init.kaiming_normal_(
            self.progress_monitor.weight, nonlinearity="tanh"
        )
        nn.init.constant_(self.progress_monitor.bias, 0)

    def forward(self, observations, rnn_states, prev_actions, masks):

        original_batch_shape = observations["original_batch_shape"].shape
        seq_length ,  batch_size = original_batch_shape
        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        instruction_embedding = self.instruction_encoder(observations)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        # the observations were "flattenened" for rnn processing
        # the first dimension is actually equall to sequence length * orginal batch size.
        # we also retrieve all other dimensions starting at index 1
        shape = lambda tensor : tuple([s for s in original_batch_shape] + [s for s in tensor.shape[1:]])

        # Transpose dimension 0 and 1 and let the last one untouched
        resize_tensor = lambda tensor: tensor.reshape(shape(tensor)).permute(1,0,-1).contiguous()

        states = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=1
        )

        #TODO init correctly form the observations
        actions = prev_actions
        returns_to_go = observations["point_nav_reward_to_go"] if self.reward_type == "POINT_GOAL_NAV_REWARD" else \
        observations["sparse_reward_to_go"]
        timesteps = observations["timesteps"].unsqueeze(-1)

        states = resize_tensor(states)
        # squeeze to output the same shape as other embeddings
        # after  the operation with embedding layer
        actions = resize_tensor(actions).squeeze()
        returns_to_go = resize_tensor(returns_to_go)
        # squeeze to output the same shape as other embeddings
        # after  the operation with embedding layer
        timesteps = resize_tensor(timesteps).squeeze()

        #The following  comes from https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py
        # https://github.com/huggingface/transformers/commit/707b12a353b69feecf11557e13d3041982bf023f


        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, -1)
        )

        stacked_inputs = self.embed_ln(stacked_inputs)
        output = self.gpt_encoder(stacked_inputs)

        # reshape back to original.
        # In the third dimension (dim=2), returns (0), states (1), or actions (2)
        # i.e. x[:,1,t] is the token for s_t
        output = output.reshape(batch_size, seq_length, 3, -1).permute(0, 2, 1, 3)

        # get predictions => TODO Actually, shouldn't we use the reward as well??
        # we should probably use  self.predict_action(output[:, 0:1]
        action_preds = self.predict_action(output[:, 1])  # predict next action given state

        return action_preds, state_embeddings
