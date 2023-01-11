import torch
import torch.nn as nn
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import BaselineRegistry
from habitat_baselines.rl.ppo.policy import Net
from vlnce_baselines.models.encoders.min_gpt import GPT
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from torch import Tensor
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
        assert net in ["DecisionTransformerNet", "DecisionTransformerEnhancedNet"]

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


class DecisionTransformerNet(Net):
    """Decision Transformer, where RGB, DEPTH and Instructions are concatenated into one state.
    Finally, a distribution over discrete actions (FWD, L, R, STOP) is produced.
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

        assert model_config.DECISION_TRANSFORMER.reward_type in ["point_nav_reward_to_go", "sparse_reward_to_go", "point_nav_reward", "sparse_reward", "ndtw_reward", "ndtw_reward_to_go"]
        self.reward_type = model_config.DECISION_TRANSFORMER.reward_type


        # size due to concatenation of instruction, depth, and rgb features
        input_state_size = self.instruction_encoder.output_size \
                           + model_config.DEPTH_ENCODER.output_size\
                            + model_config.RGB_ENCODER.output_size

        self.transformer_step_size = self.model_config.DECISION_TRANSFORMER.step_size

        self.embed_timestep = nn.Embedding(model_config.DECISION_TRANSFORMER.episode_horizon, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_return = nn.Linear(1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_state = nn.Linear(input_state_size, model_config.DECISION_TRANSFORMER.hidden_dim)
        #TODO: What do you want to use, linear or embedding? I guess it should be embedding...
        # But if you want linear you will have to modify your input entry.
        # instaed of having action 0, 1, 2 or 3 (4 if you try to add a start token), you will have a tensor of size (4,1) filled with 0 or 1.
        #  and for action 3, you will have a 1 in the 4th row and zero otherwise.
        self.embed_action = nn.Embedding(num_actions+1, model_config.DECISION_TRANSFORMER.hidden_dim)
        self.embed_ln = nn.LayerNorm(model_config.DECISION_TRANSFORMER.hidden_dim)

        self.train()

    def _flatten_batch(self, observations: Tensor, sensor_type: str):

        #quit silently
        if not sensor_type in observations.keys():
            return

        dims = observations[sensor_type].size()
        if len(dims) > 2:
            observations[sensor_type] = observations[sensor_type].view(-1, *dims[2:])


    def _create_timesteps(self,sequence_length, batch_size):

        timesteps = [torch.arange(0, sequence_length, dtype=torch.long) for _ in range(batch_size)]

        timesteps = torch.stack(timesteps, dim=0).to(self.embed_ln.weight.device)
        #timesteps = timesteps.view(-1, *timesteps.size()[2:]).unsqueeze(-1).to(self.embed_ln.weight.device)

        return timesteps

    @property
    def output_size(self):
        return self.model_config.DECISION_TRANSFORMER.hidden_dim

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
        instruction_embedding = self.instruction_encoder(observations)



        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        return instruction_embedding, depth_embedding, rgb_embedding

    def forward(self, observations, rnn_states, prev_actions, masks):


        original_batch_shape = observations["instruction"].shape[0:2]#excluding the embedding dimentions
        batch_size, seq_length = original_batch_shape

        instruction_embedding, depth_embedding, rgb_embedding = self._prepare_embeddings(observations)


        # the observations were "flattenened" for rnn processing
        # the first dimension is actually equall to sequence length * orginal batch size.
        # we also retrieve all other dimensions starting at index 1
        shape = lambda tensor : tuple([s for s in original_batch_shape] + [s for s in tensor.shape[1:]])

        # Transpose dimension 0 and 1 and let the last one untouched
        #resize_tensor = lambda tensor: tensor.reshape(shape(tensor)).permute(1,0,-1).contiguous()
        resize_tensor = lambda tensor: tensor.reshape(shape(tensor))

        states = torch.cat(
            [instruction_embedding, depth_embedding, rgb_embedding], dim=1
        )

        #TODO init correctly form the observations
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

        states = resize_tensor(states)
        # squeeze to output the same shape as other embeddings
        # after  the operation with embedding layer
        #actions = resize_tensor(actions).squeeze()
        # actions = resize_tensor(actions)
        # if len(actions.shape)> 2:
        #     actions = actions.squeeze(-1)
        # returns_to_go = resize_tensor(returns_to_go)
        # squeeze to output the same shape as other embeddings
        # after  the operation with embedding layer
        #timesteps = resize_tensor(timesteps).squeeze()
        # timesteps = resize_tensor(timesteps)
        if len(timesteps.shape)> 2:
            timesteps = timesteps.squeeze(-1)

        #The following  comes from https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py
        # https://github.com/huggingface/transformers/commit/707b12a353b69feecf11557e13d3041982bf023f


        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        #print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        state_embeddings2 = state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings


        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = (
            torch.stack((returns_embeddings2, state_embeddings2, action_embeddings2), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, self.transformer_step_size * seq_length, -1)
        )

        stacked_inputs2 = self.embed_ln(stacked_inputs)
        output = self.gpt_encoder(stacked_inputs2)

        # reshape back to original.
        # In the third dimension (dim=2), returns (0), states (1), or actions (2)
        # i.e. x[:,1,t] is the token for s_t
        output = output.reshape(batch_size, seq_length, self.transformer_step_size, -1).permute(0, 2, 1, 3)

        # get predictions => TODO Actually, shouldn't we use the reward as well??
        # we should probably use  self.predict_action(output[:, 0:1]
        #action_preds = self.predict_action(output[:, 1])  # predict next action given state
        action_preds = output[:, 1]
        #return action_preds.view(seq_length*batch_size, -1), state_embeddings

        return action_preds, state_embeddings


class DecisionTransformerEnhancedNet(Net):
    """DecisionTransformer with 3 different States Embeddings. Finally, a distribution over discrete
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

        self.model_config.defrost()
        # a step has Reward, Action, Instruction, Depth, RGB
        # the normal Decition Transformer has Instruction, Depth, RGB conctenated
        # into only one state
        self.model_config.DECISION_TRANSFORMER.step_size = 5
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
        # *3 because rgb, depth and instructions have their own reoresentation
        # in the GPT backbone
        return self.model_config.DECISION_TRANSFORMER.hidden_dim*2


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
        instruction_embedding = self.instruction_encoder(observations)

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
        original_batch_shape = observations["instruction"].shape[0:2]  # excluding the embedding dimentions
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
        depth_embedding = resize_tensor(depth_embedding)
        rgb_embedding = resize_tensor(rgb_embedding)
        if len(timesteps.shape) > 2:
            timesteps = timesteps.squeeze(-1)

        # The following  comes from https://github.com/huggingface/transformers/blob/main/src/transformers/models/decision_transformer/modeling_decision_transformer.py
        # https://github.com/huggingface/transformers/commit/707b12a353b69feecf11557e13d3041982bf023f

        # embed each modality with a different head
        instruction_state_embeddings = self.instruction_embed_state(instruction_states)
        rgb_state_embeddings = self.rgb_embed_state(rgb_embedding)
        depth_state_embeddings = self.depth_embed_state(depth_embedding)

        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # print(state_embeddings.shape, action_embeddings.shape, returns_embeddings.shape, time_embeddings.shape)
        # time embeddings are treated similar to positional embeddings
        instruction_state_embeddings2 = instruction_state_embeddings + time_embeddings
        rgb_state_embeddings2 = rgb_state_embeddings + time_embeddings
        depth_state_embeddings2 = depth_state_embeddings + time_embeddings
        action_embeddings2 = action_embeddings + time_embeddings
        returns_embeddings2 = returns_embeddings + time_embeddings



        # this makes the sequence look like (R_1, s_instr_1, s_depth_1, s_rgb_1, a_1, R_2, s_instr_2, s_depth_2, s_rgb_2, a_2, ...)
        stacked_inputs = (
            torch.stack((returns_embeddings2, instruction_state_embeddings2, rgb_state_embeddings2, depth_state_embeddings2, action_embeddings2), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, self.transformer_step_size * seq_length, -1)
        )

        stacked_inputs2 = self.embed_ln(stacked_inputs)
        output = self.gpt_encoder(stacked_inputs2)

        # reshape back to original.
        # In the third dimension (dim=2), returns (0), states (1), or actions (2)
        # i.e. x[:,1,t] is the token for s_t
        output = output.reshape(batch_size, seq_length, self.transformer_step_size, -1).permute(0, 2, 1, 3)

        # get predictions
        # we are retrieveing rgb, depth and instruction steps...
        action_preds = output[:,1:4].permute(0,2,1,3).reshape(batch_size, seq_length, -1)
        # return action_preds.view(seq_length*batch_size, -1), state_embeddings

        return action_preds, None
