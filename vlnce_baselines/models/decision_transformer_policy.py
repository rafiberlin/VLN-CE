import torch
import torch.nn as nn
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.rl.ppo.policy import Net
from vlnce_baselines.models.encoders.min_gpt import GPT, NewGELU
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder, Word2VecEmbeddings, InstructionEncoderWithTransformer
)
from vlnce_baselines.models.policy import ILPolicy

from torch import Tensor

from vlnce_baselines.models.utils import PositionalEncoding, VanillaMultiHeadAttention


@BaselineRegistry.register_policy
class DecisionTransformerPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        net = "DecisionTransformerNet"
        if hasattr(model_config.DECISION_TRANSFORMER, "net"):
            net = model_config.DECISION_TRANSFORMER.net
        assert net in ["DecisionTransformerNet",
                       "DecisionTransformerEnhancedNet",
                       "DecisionTransformerWithAttendedInstructionsNet",
                       "FullDecisionTransformerNet",
                       "FullDecisionTransformerNet2"]
        print("Training with:", net)
        super().__init__(
            eval(net)(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        actions, rnn_states = super().act(observations, rnn_states, prev_actions, masks, deterministic)
        #We just want to return the last action of the transformer sequence...
        return actions[:,-1,:], rnn_states

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


class AbstractDecisionTransformerNet(Net):
    # Decision Transformer where each time step is fed into a GPT backbone.
    # Finally, a distribution over discrete actions (FWD, L, R, STOP) is produced.
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        """

        :param observation_space: Delivered by the Habitat Framework
        :param model_config: General config
        :param num_actions: 4 discrete actions (FWD, L, R, STOP)
        """
        super().__init__()
        self.model_config = model_config
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        assert model_config.DECISION_TRANSFORMER.reward_type in ["point_nav_reward_to_go", "sparse_reward_to_go",
                                                                 "point_nav_reward", "sparse_reward", "ndtw_reward",
                                                                 "ndtw_reward_to_go"]

        n = self.initialize_transformer_step_size()
        self.set_transformer_step_size(n)
        # Init the Depth visual encoder
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
        )
        # Init the RGB visual encoder
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=False,
        )

        self.initialize_instruction_encoder()

        self.reward_type = model_config.DECISION_TRANSFORMER.reward_type
        self.action_activation = nn.Sequential(nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_action_drop), NewGELU())
        self.instruction_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_instruction_drop), NewGELU())
        self.rgb_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_rgb_drop), NewGELU())
        self.depth_activation = nn.Sequential(
            nn.Dropout(p=self.model_config.DECISION_TRANSFORMER.activation_depth_drop), NewGELU())
        self.gpt_encoder = GPT(self.model_config.DECISION_TRANSFORMER)
        self.transformer_step_size = self.model_config.DECISION_TRANSFORMER.step_size
        self.embed_timestep = nn.Embedding(model_config.DECISION_TRANSFORMER.episode_horizon, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_return = nn.Linear(1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_action = nn.Embedding(num_actions + 1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_ln = nn.LayerNorm(model_config.DECISION_TRANSFORMER.hidden_dim)
        self.initialize_other_layers()
        self.train()

    def _prepare_embeddings(self, observations):
        """
        read the relevant features from observation and returns it
        :param observations:
        :return: instruction_embedding, depth_embedding, rgb_embedding
        """
        # for all the following keys, we need tto merge the first 2 dimensions
        # [batch, sequence length, all other dimensions] to [batch * sequence length, all other dimensions]

        original_batch_shape = observations["instruction"].shape[0:2]  # excluding the embedding dimentions
        batch_size, seq_length = original_batch_shape
        # the observations were flattened for rnn processing
        # the first dimension is actually equal to sequence length * original batch size.
        # we also retrieve all other dimensions starting at index 1
        shape = lambda tensor: tuple([s for s in original_batch_shape] + [s for s in tensor.shape[1:]])

        # Transpose dimension 0 and 1 and let the last one untouched
        # resize_tensor = lambda tensor: tensor.reshape(shape(tensor)).permute(1,0,-1).contiguous()
        resize_tensor = lambda tensor: tensor.reshape(shape(tensor))

        self._flatten_batch(observations, "rgb")
        self._flatten_batch(observations, "depth")
        self._flatten_batch(observations, "rgb_features")
        self._flatten_batch(observations, "depth_features")
        if not self.model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction:
            self._flatten_batch(observations, "instruction")

        depth_embedding = self.depth_activation(self.depth_encoder(observations))
        rgb_embedding = self.rgb_activation(self.rgb_encoder(observations))
        # we just undo the permutation made in the original implementation
        instruction_embedding = self.handle_instruction_embeddings(observations, resize_tensor, batch_size, seq_length)

        depth_embedding = resize_tensor(depth_embedding)
        rgb_embedding = resize_tensor(rgb_embedding)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        return instruction_embedding, depth_embedding, rgb_embedding

    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):
        raise not NotImplementedError("Depending, if you get a sentence embedding or the whole word sequence!")

    def initialize_instruction_encoder(self):
        raise not NotImplementedError("Should set instruction encoder used by your model!")

    def initialize_other_layers(self):
        raise not NotImplementedError("Should set the layers used by your model!")

    def initialize_transformer_step_size(self):
        raise not NotImplementedError("Should return the value needed for set_transformer_step_size(self, n) ")

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                                             depth_embedding, rgb_embedding, timesteps, batch_size,
                                                             seq_length):
        raise not NotImplementedError("do the mo del specific work and return a tuple of tensors like (Action, S1, ... Sn, Reward)")

    def set_transformer_step_size(self, n):
        self.model_config.defrost()
        # a step has a size of 2 + n
        # Actions, State 1, State 2... State n, Reward
        self.model_config.DECISION_TRANSFORMER.step_size = n
        self.model_config.freeze()

    def _flatten_batch(self, observations: Tensor, sensor_type: str):

        # quit silently
        if not sensor_type in observations.keys():
            return

        dims = observations[sensor_type].size()
        if len(dims) > 2:
            observations[sensor_type] = observations[sensor_type].view(-1, *dims[2:])

    @property
    def output_size(self):
        return self.model_config.DECISION_TRANSFORMER.hidden_dim * (self.transformer_step_size - 2)# - 2 because we exclude reward / actions for categorical layer

    def create_timesteps(self, sequence_length, batch_size):

        # TODO: use buffer?
        timesteps = [torch.arange(0, sequence_length, dtype=torch.long) for _ in range(batch_size)]
        timesteps = torch.stack(timesteps, dim=0).to(self.embed_ln.weight.device)

        return timesteps

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_states, prev_actions, masks):
        original_batch_shape = observations["instruction"].shape[0:2]  # excluding the embedding dimentions
        batch_size, seq_length = original_batch_shape

        instruction_embedding, depth_embedding, rgb_embedding = self._prepare_embeddings(observations)

        if self.reward_type in observations.keys():
            returns_to_go = observations[self.reward_type]
        else:
            # If we don t have any rewards from the environment, just take one
            # as mentioned in the paper during evaluation.
            returns_to_go = torch.ones_like(prev_actions, dtype=torch.float).unsqueeze(dim=-1)
        if "timesteps" in observations.keys():
            timesteps = observations["timesteps"]
        else:
            timesteps = self.create_timesteps(seq_length, batch_size)

        # squeeze to output the same shape as other embeddings
        # after  the operation with embedding layer
        if len(timesteps.shape) > 2:
            timesteps = timesteps.squeeze(-1)

        tensor_tuples = self.create_tensors_for_gpt_as_tuple(prev_actions, returns_to_go, instruction_embedding,
                                                             depth_embedding, rgb_embedding, timesteps, batch_size,
                                                             seq_length)

        stacked = (
            torch.stack(tensor_tuples, dim=1).permute(0, 2, 1, 3).reshape(batch_size, self.transformer_step_size * seq_length, -1)
        )

        output = self.gpt_encoder(self.embed_ln(stacked))
        output = output.reshape(batch_size, seq_length, self.transformer_step_size, -1).permute(0, 2, 1, 3)

        # get predictions
        action_preds = output[:, 1:self.transformer_step_size - 1].permute(0, 2, 1, 3).reshape(batch_size, seq_length,
                                                                                               -1)

        return action_preds, None

class DecisionTransformerNet(AbstractDecisionTransformerNet):
    """Decision Transformer, where RGB, DEPTH and Instructions are concatenated into one state.
    """
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__(observation_space, model_config, num_actions)



    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):

        instruction_embedding = self.instruction_activation(self.instruction_encoder(observations))

        if not self.model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction:
            instruction_embedding = resize_tensor(instruction_embedding)
        else:
            instruction_embedding = self.sentence_encoding(instruction_embedding.permute(0, 2, 1)).permute(0, 2,
                                                                                                           1).repeat(
                (1, seq_length, 1))
        return instruction_embedding

    def initialize_other_layers(self):
        # size due to concatenation of instruction, depth, and rgb features
        input_state_size = self.instruction_encoder.output_size + self.model_config.DEPTH_ENCODER.output_size + self.model_config.RGB_ENCODER.output_size
        self.embed_state = nn.Linear(input_state_size, self.model_config.DECISION_TRANSFORMER.hidden_dim)

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        states = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=2
        )
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.action_activation(self.embed_action(prev_actions))
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings2 = state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        return returns_embeddings2, state_embeddings2, action_embeddings2

    def initialize_instruction_encoder(self):
        if not self.model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction:
            # Init the instruction encoder
            self.instruction_encoder = InstructionEncoder(
                self.model_config.INSTRUCTION_ENCODER
            )
        else:
            self.instruction_encoder = InstructionEncoderWithTransformer(self.model_config)
            if self.model_config.DECISION_TRANSFORMER.ENCODER.use_sentence_encoding:
                self.sentence_encoding = nn.AdaptiveAvgPool1d(1)

    def initialize_transformer_step_size(self):
        # action, state, reward
        return 3

class DecisionTransformerEnhancedNet(DecisionTransformerNet):

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__(observation_space, model_config, num_actions)

    def initialize_other_layers(self):
        out_dim = self.model_config.DECISION_TRANSFORMER.hidden_dim
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 out_dim)
        self.rgb_embed_state = nn.Linear(self.model_config.RGB_ENCODER.output_size,
                                         out_dim)
        self.depth_embed_state = nn.Linear(self.model_config.DEPTH_ENCODER.output_size,
                                           out_dim)

    def initialize_transformer_step_size(self):
        return 5

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                        depth_embedding, rgb_embedding, timesteps, batch_size,
                                        seq_length):
        instruction_state_embeddings = self.instruction_embed_state(instruction_embedding)
        rgb_state_embeddings = self.rgb_embed_state(rgb_embedding)
        depth_state_embeddings = self.depth_embed_state(depth_embedding)

        action_embeddings = self.embed_action(prev_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        instruction_state_embeddings2 = instruction_state_embeddings + time_embeddings
        rgb_state_embeddings2 = rgb_state_embeddings + time_embeddings
        depth_state_embeddings2 = depth_state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        return returns_embeddings2, instruction_state_embeddings2, rgb_state_embeddings2, depth_state_embeddings2, action_embeddings2


class DecisionTransformerWithAttendedInstructionsNet(Net):
    """DecisionTransformer with 3 different States Embeddings. Finally, a distribution over discrete
    actions (FWD, L, R, STOP) is produced.
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__()

        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.DECISION_TRANSFORMER.ATTENTION_LAYER.n_embd = model_config.DECISION_TRANSFORMER.hidden_dim
        model_config.freeze()

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

        self.model_config.defrost()
        # a step has Reward, Action, Instruction, Depth, RGB
        # the normal Decision Transformer has Instruction, Depth, RGB conctenated
        # into only one state
        #self.model_config.DECISION_TRANSFORMER.step_size = 8
        #self.model_config.DECISION_TRANSFORMER.step_size = 5

        self.model_config.DECISION_TRANSFORMER.step_size = 2

        if self.model_config.DECISION_TRANSFORMER.use_instruction_state_embeddings is True:
            self.model_config.DECISION_TRANSFORMER.step_size += 1
        if self.model_config.DECISION_TRANSFORMER.use_depth_influenced_text is True:
            self.model_config.DECISION_TRANSFORMER.step_size += 1
        if self.model_config.DECISION_TRANSFORMER.use_rgb_influenced_text is True:
            self.model_config.DECISION_TRANSFORMER.step_size += 1
        if self.model_config.DECISION_TRANSFORMER.use_state_attended_text is True:
            self.model_config.DECISION_TRANSFORMER.step_size += 1
        if self.model_config.DECISION_TRANSFORMER.use_rgb_state_embeddings is True:
            self.model_config.DECISION_TRANSFORMER.step_size += 1
        if self.model_config.DECISION_TRANSFORMER.use_depth_state_embeddings is True:
            self.model_config.DECISION_TRANSFORMER.step_size += 1



        self.model_config.freeze()


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

        assert model_config.DECISION_TRANSFORMER.reward_type in ["point_nav_reward_to_go", "sparse_reward_to_go",
                                                                 "point_nav_reward", "sparse_reward", "ndtw_reward",
                                                                 "ndtw_reward_to_go"]
        self.reward_type = model_config.DECISION_TRANSFORMER.reward_type

        self.embed_timestep = nn.Embedding(model_config.DECISION_TRANSFORMER.episode_horizon,
                                           model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_return = nn.Linear(1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 model_config.DECISION_TRANSFORMER.hidden_dim)
        self.rgb_embed_state = nn.Linear(model_config.RGB_ENCODER.output_size,
                                                 model_config.DECISION_TRANSFORMER.hidden_dim)
        self.depth_embed_state = nn.Linear(model_config.DEPTH_ENCODER.output_size,
                                                 model_config.DECISION_TRANSFORMER.hidden_dim)

        self.state_q_to_text = nn.Sequential(
            nn.Linear(
                self.rgb_embed_state.out_features + self.depth_embed_state.out_features,
                self.instruction_embed_state.out_features,
            ),
            nn.ReLU(True),
        )

        self.text_q_to_rgb = nn.Sequential(
            nn.Linear(
                self.instruction_embed_state.out_features,
                self.rgb_embed_state.out_features
            ),
            nn.ReLU(True),
        )

        self.text_q_to_depth = nn.Sequential(
            nn.Linear(
                self.instruction_embed_state.out_features,
                self.depth_embed_state.out_features
            ),
            nn.ReLU(True),
        )

        self.state_to_text_causal_attention = VanillaMultiHeadAttention(
            self.model_config.DECISION_TRANSFORMER.ATTENTION_LAYER)

        self.text_to_depth_attention = VanillaMultiHeadAttention(
            self.model_config.DECISION_TRANSFORMER.ATTENTION_LAYER)

        self.text_to_rgb_attention = VanillaMultiHeadAttention(
            self.model_config.DECISION_TRANSFORMER.ATTENTION_LAYER)

        # these transform the instruction sequences (originated form all rnn hidden states)
        # to a fix sentence embedding representation
        self.to_sentence_embed = nn.AdaptiveAvgPool1d(1)
        self.to_sentence_embed2 = nn.AdaptiveAvgPool1d(1)
        self.to_sentence_embed3 = nn.AdaptiveAvgPool1d(1)

        # 5 because we have embedding for reward, one for past actions and 3 for states( instructions
        # rgb and depth). In the origininal transformer, there is only one state instead of 3.
        self.transformer_step_size = self.model_config.DECISION_TRANSFORMER.step_size
        # TODO: What do you want to use, linear or embedding? I guess it should be embedding...
        # But if you want linear you will have to modify your input entry.
        # instaed of having action 0, 1, 2 or 3 (4 if you try to add a start token), you will have a tensor of size (4,1) filled with 0 or 1.
        #  and for action 3, you will have a 1 in the 4th row and zero otherwise.
        self.embed_action = nn.Embedding(num_actions + 1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_ln = nn.LayerNorm(model_config.DECISION_TRANSFORMER.hidden_dim)

        self.train()

    def _flatten_batch(self, observations: Tensor, sensor_type: str):

        # quit silently
        if not sensor_type in observations.keys():
            return

        dims = observations[sensor_type].size()
        if len(dims) > 2:
            observations[sensor_type] = observations[sensor_type].view(-1, *dims[2:])

    def _create_timesteps(self, sequence_length, batch_size):

        timesteps = [torch.arange(0, sequence_length, dtype=torch.long) for _ in range(batch_size)]

        timesteps = torch.stack(timesteps, dim=0).to(self.embed_ln.weight.device)
        # timesteps = timesteps.view(-1, *timesteps.size()[2:]).unsqueeze(-1).to(self.embed_ln.weight.device)

        return timesteps

    @property
    def output_size(self):
        # we retrieve 2 , because it corresponds to the actions an reward dimension that are removed
        # after GPT processing...
        return self.model_config.DECISION_TRANSFORMER.hidden_dim*(self.transformer_step_size -2)


    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def _prepare_embeddings(self, observations):
        """
        read the relevant features from observation and returns it
        :param observations:
        :return: instruction_embedding, depth_embedding, rgb_embedding
        """
        # for all the following keys, we need tto merge the first 2 dimensions
        # [batch, sequence length, all other dimensions] to [batch * sequence length, all other dimensions]
        self._flatten_batch(observations, "rgb")
        self._flatten_batch(observations, "depth")
        self._flatten_batch(observations, "rgb_features")
        self._flatten_batch(observations, "depth_features")
        self._flatten_batch(observations, "instruction")

        depth_embedding = self.depth_encoder(observations)
        rgb_embedding = self.rgb_encoder(observations)
        # we just undo the permutation made in the original implementation
        instruction_embedding = self.instruction_encoder(observations).permute(0, 2, 1)

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        return instruction_embedding, depth_embedding, rgb_embedding

    def forward(self, observations, rnn_states, prev_actions, masks):

        # need to be checked now as the function _prepare_embeddings() modifies the
        # obsevations directly.
        original_batch_shape = observations["instruction"].shape[0:2]  # excluding the embedding dimensions
        batch_size, seq_length = original_batch_shape

        instruction_embedding, depth_embedding, rgb_embedding = self._prepare_embeddings(observations)


        # the observations were "flattenened" for rnn processing
        # the first dimension is actually equall to sequence length * orginal batch size.
        # we also retrieve all other dimensions starting at index 1
        shape = lambda tensor: tuple([s for s in original_batch_shape] + [s for s in tensor.shape[1:]])

        # Transpose dimension 0 and 1 and let the last one untouched
        # resize_tensor = lambda tensor: tensor.reshape(shape(tensor)).permute(1,0,-1).contiguous()
        resize_tensor = lambda tensor: tensor.reshape(shape(tensor))



        # TODO init correctly form the observations
        actions = prev_actions

        if self.reward_type in observations.keys():
            returns_to_go = observations[self.reward_type]
        else:
            # If we don t have any rewards from the environment, just take one
            # as mentioned in the paper during evaluation.
            returns_to_go = torch.ones_like(prev_actions, dtype=torch.float).unsqueeze(dim=-1)
        if "timesteps" in observations.keys():
            timesteps = observations["timesteps"]
        else:
            timesteps = self._create_timesteps(seq_length, batch_size)

        instruction_states = resize_tensor(instruction_embedding)
        # The instructions are repeated for each time step.
        # here we just want to one representation per batch
        single_instruction_states = instruction_states[:, 0, :, :]
        depth_embedding = resize_tensor(depth_embedding)
        rgb_embedding = resize_tensor(rgb_embedding)
        if len(timesteps.shape) > 2:
            timesteps = timesteps.squeeze(-1)

        # The following  comes from https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py
        # https://github.com/huggingface/transformers/commit/707b12a353b69feecf11557e13d3041982bf023f

        # embed each modality with a different head
        #WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Will probably mess up with dimensonality with the timesteps
        instruction_state_embeddings = self.instruction_embed_state(single_instruction_states)
        rgb_state_embeddings = self.rgb_embed_state(rgb_embedding)
        depth_state_embeddings = self.depth_embed_state(depth_embedding)

        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # needs to add the time embeddings as the attention layer I use do not add it automaticallys
        concat_state_inputs = torch.cat((rgb_state_embeddings + time_embeddings, depth_state_embeddings + time_embeddings), dim=-1)
        state_q_to_text = self.state_q_to_text(concat_state_inputs)
        causal_mask = VanillaMultiHeadAttention.create_causal_mask(seq_length, state_q_to_text.device)
        text_mask = VanillaMultiHeadAttention.create_padded_mask(single_instruction_states)
        state_q_to_text = self.state_to_text_causal_attention(state_q_to_text, state_q_to_text, state_q_to_text, causal_mask)

        state_attended_text = self.state_to_text_causal_attention(q=state_q_to_text, v=instruction_state_embeddings, k=instruction_state_embeddings, mask=text_mask)

        # WARNING Maybe you don't need a key for RGB and one for DEPTH, it might get reused?
        # Should we also use a padded mask as well?
        text_q_to_depth = self.text_q_to_depth(instruction_state_embeddings)
        text_attended_depth = self.text_to_depth_attention(q=text_q_to_depth, k=depth_state_embeddings, v=depth_state_embeddings)
        text_q_to_rgb = self.text_q_to_rgb(instruction_state_embeddings)
        text_attended_key = self.text_to_rgb_attention(q=text_q_to_rgb, k=rgb_state_embeddings, v=rgb_state_embeddings)

        # text_attended_*** have a shape of Batch*Instruction Length* Dim => the permutation allows to get only one embedding for the whole instructions
        # TODO ITHIN THE POOLING has no weight. Hence, instead of reducing it to one representation, you can create
        # as many as time steps!!!
        depth_influenced_text = self.to_sentence_embed(text_attended_depth.permute(0,2,1)).permute(0,2,1)
        rgb_influenced_text = self.to_sentence_embed2(text_attended_key.permute(0,2,1)).permute(0,2,1)
        instruction_sentence_embeddings = self.to_sentence_embed2(instruction_state_embeddings.permute(0, 2, 1)).permute(0, 2, 1)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        instruction_state_embeddings2 = instruction_sentence_embeddings + time_embeddings
        depth_influenced_text2 = depth_influenced_text + time_embeddings
        rgb_influenced_text2 = rgb_influenced_text + time_embeddings
        state_attended_text2 = state_attended_text + time_embeddings # Here maybe you shouldn t add the time embeddings? As you already did it in the
        rgb_state_embeddings2 = rgb_state_embeddings + time_embeddings
        depth_state_embeddings2 = depth_state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        dim_for_concat = 1
        stacked = returns_embeddings2.unsqueeze(dim_for_concat)

        if self.model_config.DECISION_TRANSFORMER.use_instruction_state_embeddings is True:
            stacked = torch.cat((stacked, instruction_state_embeddings2.unsqueeze(dim_for_concat)), dim=dim_for_concat)
        if self.model_config.DECISION_TRANSFORMER.use_depth_influenced_text is True:
            stacked = torch.cat((stacked, rgb_state_embeddings2.unsqueeze(dim_for_concat)), dim=dim_for_concat)
        if self.model_config.DECISION_TRANSFORMER.use_rgb_influenced_text is True:
            stacked = torch.cat((stacked, depth_state_embeddings2.unsqueeze(dim_for_concat)), dim=dim_for_concat)
        if self.model_config.DECISION_TRANSFORMER.use_state_attended_text is True:
            stacked = torch.cat((stacked, depth_influenced_text2.unsqueeze(dim_for_concat)), dim=dim_for_concat)
        if self.model_config.DECISION_TRANSFORMER.use_rgb_state_embeddings is True:
            stacked = torch.cat((stacked, rgb_influenced_text2.unsqueeze(dim_for_concat)), dim=dim_for_concat)
        if self.model_config.DECISION_TRANSFORMER.use_depth_state_embeddings is True:
            stacked = torch.cat((stacked, action_embeddings2.unsqueeze(dim_for_concat)), dim=dim_for_concat)

        stacked = torch.cat((stacked, state_attended_text2.unsqueeze(dim_for_concat)), dim=dim_for_concat)

        # this makes the sequence look like (R_1, s_instr_1, s_depth_1, s_rgb_1, a_1, R_2, s_instr_2, s_depth_2, s_rgb_2, a_2, ...)
        stacked_inputs = (
            stacked.permute(0, 2, 1, 3).reshape(batch_size, self.transformer_step_size * seq_length, -1)
        )

        stacked_inputs2 = self.embed_ln(stacked_inputs)
        output = self.gpt_encoder(stacked_inputs2)

        # reshape back to original.
        # In the third dimension (dim=2), returns (0), states (1), or actions (2)
        # i.e. x[:,1,t] is the token for s_t
        output = output.reshape(batch_size, seq_length, self.transformer_step_size, -1).permute(0, 2, 1, 3)

        # get predictions
        # we are retrieveing rgb, depth and instruction steps...
        # WARNING: Wrong SLICE?
        action_preds = output[:,1:self.transformer_step_size - 1].permute(0,2,1,3).reshape(batch_size, seq_length, -1)
        # return action_preds.view(seq_length*batch_size, -1), state_embeddings

        return action_preds, None

class FullDecisionTransformerNet(AbstractDecisionTransformerNet):
    """DecisionTransformer with 3 different States Embeddings. Finally, a distribution over discrete
    actions (FWD, L, R, STOP) is produced.
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        # We do use Transformer encoding, but it is done to force to flatten the entry for instructions.
        model_config.DECISION_TRANSFORMER.use_transformer_encoded_instruction = False
        model_config.freeze()
        super().__init__(observation_space, model_config, num_actions)


    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):
        instruction_embedding = self.instruction_encoder(observations).permute(0, 2, 1)
        instruction_embedding = resize_tensor(instruction_embedding)
        return  instruction_embedding


    def initialize_instruction_encoder(self):
        self.instruction_encoder = Word2VecEmbeddings(
            self.model_config.INSTRUCTION_ENCODER
        )

    def initialize_other_layers(self):
        self.positional_encoding_for_instruction = PositionalEncoding(self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.encoder_instruction_to_state = nn.Transformer(d_model=self.model_config.DECISION_TRANSFORMER.hidden_dim
                                                           , nhead=self.model_config.DECISION_TRANSFORMER.n_head
                                                           ,
                                                           num_encoder_layers=self.model_config.DECISION_TRANSFORMER.ENCODER.n_layer
                                                           , num_decoder_layers=self.model_config.DECISION_TRANSFORMER.n_layer
                                                           , dim_feedforward=self.model_config.DECISION_TRANSFORMER.hidden_dim * 2
                                                           , activation="gelu"
                                                           , batch_first=True)
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.rgb_embed_state = nn.Linear(self.model_config.RGB_ENCODER.output_size,
                                         self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.depth_embed_state = nn.Linear(self.model_config.DEPTH_ENCODER.output_size,
                                           self.model_config.DECISION_TRANSFORMER.hidden_dim)

    def initialize_transformer_step_size(self):
        return 6

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                                             depth_embedding, rgb_embedding, timesteps, batch_size,
                                                             seq_length):

        single_instruction_states = instruction_embedding[:, 0, :, :]
        instruction_state_embeddings = self.positional_encoding_for_instruction(
            self.instruction_embed_state(single_instruction_states.permute(0, 2, 1)))

        step_size = 2

        # only 2D allowed in Pytorch Inmplementation
        causal_mask = VanillaMultiHeadAttention.create_causal_mask(seq_length * step_size, rgb_embedding.device)[0][0]
        text_mask = VanillaMultiHeadAttention.create_padded_mask(instruction_state_embeddings)

        rgb_state_embeddings = self.rgb_embed_state(rgb_embedding)
        depth_state_embeddings = self.depth_embed_state(depth_embedding)

        action_embeddings = self.embed_action(prev_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        rgb_state_embeddings2 = rgb_state_embeddings + time_embeddings
        depth_state_embeddings2 = depth_state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_instr_1, s_depth_1, s_rgb_1, a_1, R_2, s_instr_2, s_depth_2, s_rgb_2, a_2, ...)
        vision_inputs = (
            torch.stack((rgb_state_embeddings2, depth_state_embeddings2), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, step_size * seq_length, -1)
        )
        vision_inputs = self.embed_ln(vision_inputs)

        output = self.encoder_instruction_to_state(src=instruction_state_embeddings, tgt=vision_inputs,
                                                   src_key_padding_mask=text_mask, tgt_mask=causal_mask)

        # reshape back to original.
        # it contains instructions seen by depth and instructions seen by rgb...
        new_instructions = output.reshape(batch_size, seq_length, step_size, -1).permute(0, 2, 1, 3)

        return returns_embeddings2, rgb_state_embeddings2, depth_state_embeddings2, new_instructions[:,0], new_instructions[:,1], action_embeddings2

class FullDecisionTransformerNet2(FullDecisionTransformerNet):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ):
        super().__init__(observation_space, model_config, num_actions)

    def handle_instruction_embeddings(self, observations, resize_tensor, batch_size, seq_length):
        instruction_embedding = self.instruction_encoder(observations).permute(0, 2, 1)
        instruction_embedding = resize_tensor(instruction_embedding)
        return  instruction_embedding


    def initialize_instruction_encoder(self):
        self.instruction_encoder = Word2VecEmbeddings(
            self.model_config.INSTRUCTION_ENCODER
        )

    def initialize_other_layers(self):
        self.positional_encoding_for_instruction = PositionalEncoding(self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.instruction_embed_state = nn.Linear(self.instruction_encoder.output_size,
                                                 self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.rgb_embed_state = nn.Linear(self.model_config.RGB_ENCODER.output_size,
                                         self.model_config.DECISION_TRANSFORMER.hidden_dim)
        self.depth_embed_state = nn.Linear(self.model_config.DEPTH_ENCODER.output_size,
                                           self.model_config.DECISION_TRANSFORMER.hidden_dim)

        def prepare_transformer_layer(model_config):
            return nn.Transformer(d_model=model_config.DECISION_TRANSFORMER.hidden_dim
                                               , nhead=model_config.DECISION_TRANSFORMER.n_head
                                               , num_encoder_layers=model_config.DECISION_TRANSFORMER.ENCODER.n_layer
                                               , num_decoder_layers=model_config.DECISION_TRANSFORMER.n_layer
                                               , dim_feedforward=model_config.DECISION_TRANSFORMER.hidden_dim * 2
                                               , activation="gelu"
                                               , batch_first=True)

        self.encoder_instruction_to_state = prepare_transformer_layer(self.model_config)
        self.encoder_rgb_to_instruction = prepare_transformer_layer(self.model_config)
        self.encoder_depth_to_instruction = prepare_transformer_layer(self.model_config)
        self.visual_to_sentence_embed = nn.AdaptiveAvgPool1d(1)

    def initialize_transformer_step_size(self):

        return 8

    def create_tensors_for_gpt_as_tuple(self, prev_actions, returns_to_go, instruction_embedding,
                                                             depth_embedding, rgb_embedding, timesteps, batch_size,
                                                             seq_length):
        single_instruction_states = instruction_embedding[:, 0, :, :]
        # embed each modality with a different head
        instruction_state_embeddings = self.positional_encoding_for_instruction(
            self.instruction_embed_state(single_instruction_states.permute(0, 2, 1)))

        step_size = 2
        # only 2D allowed in Pytorch Inmplementation
        causal_mask = VanillaMultiHeadAttention.create_causal_mask(seq_length * step_size, rgb_embedding.device)[0][0]
        text_mask = VanillaMultiHeadAttention.create_padded_mask(instruction_state_embeddings)

        rgb_state_embeddings = self.rgb_embed_state(rgb_embedding)
        depth_state_embeddings = self.depth_embed_state(depth_embedding)

        action_embeddings = self.embed_action(prev_actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        rgb_state_embeddings2 = rgb_state_embeddings + time_embeddings
        depth_state_embeddings2 = depth_state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_instr_1, s_depth_1, s_rgb_1, a_1, R_2, s_instr_2, s_depth_2, s_rgb_2, a_2, ...)
        stacked_inputs = (
            torch.stack((rgb_state_embeddings2, depth_state_embeddings2), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, step_size * seq_length, -1)
        )
        # torch.stack((returns_embeddings2, instruction_state_embeddings2, rgb_state_embeddings2, depth_state_embeddings2, depth_influenced_text2, rgb_influenced_text2, state_attended_text2, action_embeddings2), dim=1)
        stacked_inputs2 = self.embed_ln(stacked_inputs)

        causal_text_mask = VanillaMultiHeadAttention.create_causal_mask(instruction_state_embeddings.shape[1],
                                                                        instruction_state_embeddings.device)[0][0]
        rgb_mask = VanillaMultiHeadAttention.create_padded_mask(rgb_state_embeddings2)
        depth_mask = VanillaMultiHeadAttention.create_padded_mask(depth_state_embeddings2)
        output_instructions = self.encoder_instruction_to_state(src=instruction_state_embeddings, tgt=stacked_inputs2,
                                                   src_key_padding_mask=text_mask, tgt_mask=causal_mask)
        output_rgb = self.encoder_rgb_to_instruction(src=rgb_state_embeddings2, tgt=instruction_state_embeddings,
                                                     src_key_padding_mask=rgb_mask, tgt_mask=causal_text_mask)
        output_depth = self.encoder_depth_to_instruction(src=depth_state_embeddings2, tgt=instruction_state_embeddings,
                                                         src_key_padding_mask=depth_mask, tgt_mask=causal_text_mask)

        output_rgb = self.visual_to_sentence_embed(output_rgb.permute(0, 2, 1)).permute(0, 2, 1) + time_embeddings
        output_depth = self.visual_to_sentence_embed(output_depth.permute(0, 2, 1)).permute(0, 2, 1) + time_embeddings

        # reshape back to original.
        output_instructions = output_instructions.reshape(batch_size, seq_length, step_size, -1).permute(0, 2, 1, 3)

        return returns_embeddings2, rgb_state_embeddings2, depth_state_embeddings2, output_instructions[:,0], output_instructions[:,1],output_rgb, output_depth,  action_embeddings2

