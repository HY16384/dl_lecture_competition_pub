import torch
from torch import nn
from src.models.base import *
from src.models.raft_base import *
from src.utils import *
from typing import Dict, Any

'''
参考: https://pytorch.org/vision/0.12/_modules/torchvision/models/optical_flow/raft.html
'''

_BASE_CHANNELS = 64

class RAFT(nn.Module):
    def __init__(self, feature_encoder, context_encoder, corr_block, update_block, mask_predictor=None):
        super(RAFT,self).__init__()

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.corr_block = corr_block
        self.update_block = update_block

        self.mask_predictor = mask_predictor

    def forward(self, inputs_now: Dict[str, Any], inputs_prev: Dict[str, Any], num_flow_updates: int = 12) -> Dict[str, Any]:
        batch_size, _, h, w = inputs_prev.shape

        fmaps = self.feature_encoder(torch.cat([inputs_now, inputs_prev], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        self.corr_block.build_pyramid(fmap1, fmap2)

        context_out = self.context_encoder(inputs_now)
        
        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[1] - hidden_state_size

        #隠れ状態の初期値をcontextをもとにして決定している
        hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
        hidden_state = torch.tanh(hidden_state) 
        context = F.relu(context)

        coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)

        flow_predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach() # Don't backpropagate gradients through this branch, see paper とかかれていた、なんでなんでしょうか？
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)
            flow = coords1 - coords0

            hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)

            coords1 = coords1 + delta_flow

            up_mask = self.mask_predictor(hidden_state)
            upsampled_flow = upsample_flow(flow=(coords1 - coords0), up_mask=up_mask) #元のサイズに変換

            flow_predictions.append(upsampled_flow)

        return flow_predictions
    
#パラメータ初期化
def _raft(
    # Feature encoder
    feature_encoder_layers,
    # Context encoder
    context_encoder_layers,
    # Correlation block
    corr_block_num_levels,
    corr_block_radius,
    # Motion encoder
    motion_encoder_corr_layers,
    motion_encoder_flow_layers,
    motion_encoder_out_channels,
    # Recurrent block
    recurrent_block_hidden_state_size,
    # Flow Head
    flow_head_hidden_size,
    **kwargs,
):
    feature_encoder = kwargs.pop("feature_encoder", None) or FeatureEncoder(4, res_layers=feature_encoder_layers)
    context_encoder = kwargs.pop("context_encoder", None) or FeatureEncoder(4, res_layers=context_encoder_layers)

    corr_block = kwargs.pop("corr_block", None) or CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)

    update_block = kwargs.pop("update_block", None)
    if update_block is None:
        motion_encoder = MotionEncoder(
            in_channels_corr=corr_block.out_channels,
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels,
        )

    out_channels_context = context_encoder_layers[-1] - recurrent_block_hidden_state_size
    recurrent_block = RecurrentBlock(
        input_size=motion_encoder.out_channels + out_channels_context,
        hidden_size=recurrent_block_hidden_state_size,
    )

    flow_head = FlowHead(
        in_channels=recurrent_block_hidden_state_size,
        hidden_size=flow_head_hidden_size
    )
    update_block = UpdateBlock(
        motion_encoder=motion_encoder,
        recurrent_block=recurrent_block,
        flow_head=flow_head
    )

    mask_predictor = kwargs.pop("mask_predictor", None)
    if mask_predictor is None:
        mask_predictor = MaskPredictor(
            in_channels=recurrent_block_hidden_state_size,
            hidden_size=256,
            multiplier=0.25,  # See comment in MaskPredictor about this
        )

    model = RAFT(
        feature_encoder=feature_encoder,
        context_encoder=context_encoder,
        corr_block=corr_block,
        update_block=update_block,
        mask_predictor=mask_predictor
    )

    return model

def get_raft_model(**kwargs):
    return _raft(
        # Feature encoder
        feature_encoder_layers=(32, 64, 96, 128),
        # Context encoder
        context_encoder_layers=(32, 64, 96, 128),
        # Correlation block
        corr_block_num_levels=4,
        corr_block_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(96),
        motion_encoder_flow_layers=(64, 32),
        motion_encoder_out_channels=82,
        # Recurrent block
        recurrent_block_hidden_state_size=96,
        # Flow head
        flow_head_hidden_size=128,
        **kwargs,
    )