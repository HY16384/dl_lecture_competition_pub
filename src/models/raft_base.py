import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.models.base import *
from typing import List
from src.utils import *

'''
参考: https://pytorch.org/vision/0.12/_modules/torchvision/models/optical_flow/raft.html
'''

class MaskPredictor(nn.Module):
    """
    upsamplingをする層
    """

    def __init__(self, *, in_channels, hidden_size, multiplier=0.25):
        super().__init__()
        self.convrelu = general_conv2d(in_channels, hidden_size, do_batch_norm=False)
        # 8 * 8 * 9 because the predicted flow is downsampled by 8, from the downsampling of the initial FeatureEncoder
        # and we interpolate with all 9 surrounding neighbors. See paper and appendix B.
        self.conv = nn.Conv2d(hidden_size, 8 * 8 * 9, 1)

        # In the original code, they use a factor of 0.25 to "downweight the gradients" of that branch.
        # See e.g. https://github.com/princeton-vl/RAFT/issues/119#issuecomment-953950419
        # or https://github.com/princeton-vl/RAFT/issues/24.
        # It doesn't seem to affect epe significantly and can likely be set to 1.
        # らしい
        self.multiplier = multiplier

    def forward(self, x):
        x = self.convrelu(x)
        x = self.conv(x)
        return self.multiplier * x

class UpdateBlock(nn.Module):
    """
    flowを正解に近づけていく層
    motion encoderとrecurrent block、flow headを持つ
    """
    def __init__(self, motion_encoder, recurrent_block, flow_head):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.recurrent_block = recurrent_block
        self.flow_head = flow_head

        self.hidden_state_size = recurrent_block.hidden_size
    
    def forward(self, hidden_state, context, corr_features, flow):
        motion_features = self.motion_encoder(flow, corr_features)
        x = torch.cat([context, motion_features], dim=1)
        hidden_state = self.recurrent_block(hidden_state, x)
        delta_flow = self.flow_head(hidden_state)
        return hidden_state, delta_flow

class FlowHead(nn.Module):
    """
    recurrent blockからの隠れ状態からflowの修正分を算出する層
    """

    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class RecurrentBlock(nn.Module):
    """
    再帰的に更新していく層
    隠れ状態を受け取ってそれを次の層に出力していくのは普通のRNNと同じ
    普通の入力としては、motion encoderの出力とcontexを結合したものを受け取る
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.convgru1 = ConvGRU(input_size=input_size, hidden_size=hidden_size, kernel_size=3, padding=1)
        # self.convgru2 = ConvGRU(input_size=input_size, hidden_size=hidden_size, kernel_size=5, padding=2)
        self.hidden_size = hidden_size
    
    def forward(self, h, x):
        h = self.convgru1(h, x)
        # h = self.convgru2(h, x)
        return h

class ConvGRU(nn.Module):
    """
    Covolutional GRU
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.convz = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h

class MotionEncoder(nn.Module):
    '''
    flowの予測と、相関取ったやつを入力として受け取って、出力を出す
    出力はRecurrentBlockの入力になる
    '''

    def __init__(self, in_channels_corr, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128):
        super(MotionEncoder,self).__init__()

        self.convcorr1 = general_conv2d(in_channels=in_channels_corr, out_channels=corr_layers, ksize=1, strides=1, do_batch_norm=False)
        # self.convcorr2 = general_conv2d(in_channels=corr_layers[0], out_channels=corr_layers[1], do_batch_norm=False)

        self.convflow1 = general_conv2d(in_channels=2, out_channels=flow_layers[0], do_batch_norm=False, ksize=7, strides=1)
        self.convflow2 = general_conv2d(in_channels=flow_layers[0], out_channels=flow_layers[1], strides=1, do_batch_norm=False)

        #corrとflowを結合したものを入力にして出力する outchannelを-2しているのは、後からflowを結合するから
        self.conv = general_conv2d(in_channels=corr_layers+flow_layers[-1], out_channels=out_channels-2, strides=1, do_batch_norm=False)
        
        self.out_channels = out_channels

    def forward(self, flow, corr_features):
        corr = self.convcorr1(corr_features)
        # corr = self.convcorr2(corr)

        flow_original = flow
        flow = self.convflow1(flow)
        flow = self.convflow2(flow)

        corr_flow = torch.cat([corr, flow], dim=1)
        corr_flow = self.conv(corr_flow)

        return torch.cat([corr_flow, flow_original], dim=1)

class CorrBlock(nn.Module):
    """
    二つの入力からそれらのピクセル同士の相関を考えて、出力を作り出す
    複数の粒度で関係性を見る(average_poolを使う)
    radius内のピクセルを一つのリストにまとめたりもしている(?らしい)
    """

    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid: List[Tensor] = [torch.tensor(0)]

        self.out_channels = num_levels * (2 * radius + 1) ** 2

    #さまざまな粒度での、2つのフレームの関係性を調べる
    def build_pyramid(self, fmap1, fmap2):
        corr_volume = self._compute_corr_volume(fmap1, fmap2)
        batch_size, h, w, num_channels, _, _ = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w) #<-なんで?
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    #よく分からない
    def index_pyramid(self, centroids_coords):
        neighborhood_side_len = 2 * self.radius + 1
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2) #<-???

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(
                batch_size, h, w, -1
            )
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords /= 2

            corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()
            
            expected_output_shape = (batch_size, self.out_channels, h, w)
            torch._assert(
                corr_features.shape == expected_output_shape,
                f"Output shape of index pyramid is incorrect. Should be {expected_output_shape}, got {corr_features.shape}",
            )

        return corr_features


    #相関をとる（それぞれのピクセルごとに内積を取って、チャンネル数のルートで割っている）
    def _compute_corr_volume(self, fmap1, fmap2):
        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h * w)
        fmap2 = fmap2.view(batch_size, num_channels, h * w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch_size, h, w, 1, h, w)

        return corr / torch.sqrt(torch.tensor(num_channels))

class FeatureEncoder(nn.Module):
    """
    特徴を抽出する feature encoderにもcontext encoderにも使われる
    サイズが-8になる(らしい)点に注意
    """

    def __init__(self, in_channels, res_layers=(64, 64, 96, 128), out_channels=256, do_batch_norm=True):
        super(FeatureEncoder,self).__init__()

        assert len(res_layers) == 4

        self.conv1 = general_conv2d(in_channels=in_channels, out_channels=res_layers[0], ksize=7, strides=2, do_batch_norm=do_batch_norm)
        self.layer1 = nn.Sequential(
            build_bottleneck_block(in_channels=res_layers[0], out_channels=res_layers[1], do_batch_norm=do_batch_norm),
            build_bottleneck_block(in_channels=res_layers[1], out_channels=res_layers[1], do_batch_norm=do_batch_norm)
            # build_resnet_block(in_channels=res_layers[0], out_channels=res_layers[1], do_batch_norm=do_batch_norm),
            # build_resnet_block(in_channels=res_layers[1], out_channels=res_layers[1], do_batch_norm=do_batch_norm)
        )
        self.layer2 = nn.Sequential(
            build_bottleneck_block(in_channels=res_layers[1], out_channels=res_layers[2], do_batch_norm=do_batch_norm),
            build_bottleneck_block(in_channels=res_layers[2], out_channels=res_layers[2], do_batch_norm=do_batch_norm)
            # build_resnet_block(in_channels=res_layers[1], out_channels=res_layers[2], do_batch_norm=do_batch_norm),
            # build_resnet_block(in_channels=res_layers[2], out_channels=res_layers[2], do_batch_norm=do_batch_norm)
        )
        self.layer3 = nn.Sequential(
            build_bottleneck_block(in_channels=res_layers[2], out_channels=res_layers[3], do_batch_norm=do_batch_norm),
            build_bottleneck_block(in_channels=res_layers[3], out_channels=res_layers[3], do_batch_norm=do_batch_norm)
            # build_resnet_block(in_channels=res_layers[2], out_channels=res_layers[3], do_batch_norm=do_batch_norm),
            # build_resnet_block(in_channels=res_layers[3], out_channels=res_layers[3], do_batch_norm=do_batch_norm)
        )
        self.conv2 = general_conv2d(in_channels=res_layers[3], out_channels=out_channels, ksize=1, strides=1,)

    def forward(self, input):
        input = self.conv1(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.conv2(input)
        return input