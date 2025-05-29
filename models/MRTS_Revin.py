import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN
from typing import Callable, Optional
from torch import Tensor
from layers.PatchTST_layers import *
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from torch.utils.hooks import RemovableHandle

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars  ,patch_num,d_model]
        x=torch.reshape(x,(-1,self.n_vars,x.shape[-1],x.shape[-2]))# x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn)

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=128,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask,
                                                         attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # print(d_model, d_ff, bias)
        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
                                                attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src

class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class layer_block(nn.Module):
    def __init__(self, c_in, c_out, k_size):
        super(layer_block, self).__init__()
        self.conv_output = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 2))

        # self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1), padding=(0, int( (k_size-1)/2 ) ) )
        # self.output = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.conv_output1 = nn.Conv2d(c_in, c_out, kernel_size=(1, k_size), stride=(1, 1))
        self.output = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.relu = nn.ReLU()

    def forward(self, input):
        conv_output = self.conv_output(input)  # shape (B, D, N, T)

        conv_output1 = self.conv_output1(input)
        # print(conv_output1.size())
        output = self.output(conv_output1)
        # print(output.size())
        return self.relu(output + conv_output[..., -output.shape[3]:])

        # return self.relu( conv_output )


class multi_scale_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, seq_length, layer_num, kernel_set, layer_norm_affline=True):
        super(multi_scale_block, self).__init__()

        self.seq_length = seq_length
        self.layer_num = layer_num
        self.norm = nn.ModuleList()
        self.scale = nn.ModuleList()

        for i in range(self.layer_num):
            self.norm.append(nn.BatchNorm2d(c_out, affine=False))

        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1))
        # self.start_conv = nn.Conv1d(c_in)
        self.scale.append(nn.Conv2d(c_out, c_out, kernel_size=(1, kernel_set[0]), stride=(1, 1)))

        for i in range(1, self.layer_num):
            self.scale.append(layer_block(c_out, c_out, kernel_set[i]))

    def forward(self, input):  # input shape: B D N T



        scale = []
        scale_temp = input

        scale_temp = self.start_conv(scale_temp)
        # scale.append(scale_temp)
        for i in range(self.layer_num):
            scale_temp = self.scale[i](scale_temp)
            # scale_temp = self.norm[i](scale_temp)
            # scale_temp = self.norm[i](scale_temp, self.idx)

            # scale.append(scale_temp[...,-self.k:])
            scale.append(scale_temp)

        return scale


class DML(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(DML, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        # self.c_in=configs.enc_in
        self.revin = False
        if self.revin:
            self.revin_layer = RevIN(self.channels, affine=True, subtract_last=False)

        self.decomposition=True
        in_dim=1
        conv_channels=1
        layer_num = 3
        self.kernel_set=[7, 6, 3,2]
        self.layer_num=layer_num
        self.multi_scale_block_season = multi_scale_block(in_dim, conv_channels, self.channels, self.seq_len, layer_num,
                                                   self.kernel_set)
        self.multi_scale_block_trend = multi_scale_block(in_dim, conv_channels, self.channels, self.seq_len, layer_num,
                                                          self.kernel_set)

        scale_len=[]
        scale_len.append(self.seq_len-self.kernel_set[0]+1)
        for i in range(1,layer_num):
            # scale_len_item=(scale_len[i-1]-self.kernel_set[i]+1-3+1)//2
            scale_len_item=int((scale_len[i - 1] - self.kernel_set[i]) / 2)
            scale_len.append(scale_len_item)

        self.Linear_Seasonal=nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        for i in  range(layer_num):
            self.Linear_Seasonal.append(nn.Linear(scale_len[i],self.pred_len))
            self.Linear_Trend.append(nn.Linear(scale_len[i],self.pred_len))
        self.scale_len=scale_len


    def forward(self, x,batch_x_mark, dec_inp, batch_y_mark, batch_y=None):
        # x: [Batch, Input length, Channel]
        # print(self.scale_len)
        # 分解
        if self.decomposition:
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        #revin 标准化+仿射变换
            if self.revin:
                seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
                seasonal_init=self.revin_layer(seasonal_init,'norm')
                trend_init = self.revin_layer(trend_init, 'norm')
                seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

            # seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            seasonal_init=torch.unsqueeze(seasonal_init,dim=1)
            trend_init=torch.unsqueeze(trend_init,dim=1)
            seasonal_scale = self.multi_scale_block_season(seasonal_init)
            trend_scale=self.multi_scale_block_trend(trend_init)

            # print(seasonal_scale[0].size(), seasonal_scale[1].size(), seasonal_scale[2].size())
            out_season_lsit=[]
            out_trend_list=[]
            for i in range(self.layer_num):
                out_season=self.Linear_Seasonal[i](seasonal_scale[i])
                out_trend=self.Linear_Trend[i](trend_scale[i])
                out_season_lsit.append(out_season)
                out_trend_list.append(out_trend)

            season= sum(out_season_lsit)
            trend=sum(out_trend_list)
            season=torch.squeeze(season,dim=1)
            trend = torch.squeeze(trend, dim=1)
            # print("season,trend",season.shape,trend.shape)
            if self.revin:
                season, trend = season.permute(0, 2, 1), trend.permute(0, 2, 1)
                season=self.revin_layer(season,'denorm')
                trend = self.revin_layer(trend, 'denorm')
                season, trend = season.permute(0, 2, 1), trend.permute(0, 2, 1)
            x=season+trend
            return x.permute(0,2,1) # to [Batch, Output length, Channel]

        else:
            x = x.permute(0, 2, 1)
            if self.revin:
                x = x.permute(0, 2, 1)
                # print(x.shape)
                x = self.revin_layer(x, 'norm')
                x = x.permute(0, 2, 1)

            # seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            x = torch.unsqueeze(x, dim=1)
            # trend_init = torch.unsqueeze(trend_init, dim=1)
            x_scale = self.multi_scale_block_season(x)
            # trend_scale = self.multi_scale_block_trend(trend_init)

            # print(seasonal_scale[0].size(), seasonal_scale[1].size(), seasonal_scale[2].size())
            x_lsit = []
            # out_trend_list = []
            for i in range(self.layer_num):
                print("in",x_scale[i].size())
                out_x = self.Linear_Seasonal[i](x_scale[i])
                print("out",out_x.size())
                # out_trend = self.Linear_Trend[i](trend_scale[i])
                x_lsit.append(out_x)
                # out_trend_list.append(out_trend)

            x = sum(x_lsit)
            # trend = sum(out_trend_list)
            x = torch.squeeze(x, dim=1)
            # trend = torch.squeeze(trend, dim=1)
            # print("season,trend",season.shape,trend.shape)
            if self.revin:
                x = x.permute(0, 2, 1)
                x = self.revin_layer(x, 'denorm')
                # trend = self.revin_layer(trend, 'denorm')
                x = x.permute(0, 2, 1)
            # x = season + trend
            return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]

from typing import Callable




# from .utils import get_activation


def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()

    raise RuntimeError("activation should not be {}".format(activ))


class MLPBlock(nn.Module):

    def __init__(
        self,
        dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='trunc',
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            DropPath(drop))
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == 'proj':
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x)[..., :self.out_features] + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x


class PatchEncoder(nn.Module):

    def __init__(
        self,
        in_len: int,
        hid_len: int,
        in_chn: int,
        hid_chn: int,
        out_chn,
        patch_size: int,
        hid_pch: int,
        d_model:int,
        patch_num:int,
        nhead:int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        # self.net = nn.Sequential()
        self.net =[]
        self.net.append(Rearrange("b c (l1 l2) -> b c l1 l2", l2=patch_size))
        self.net.append(nn.Linear(patch_size, d_model))
        self.net.append(Rearrange("b c l1 l2 -> (b c) l1 l2"))
        self.net.append(TSTEncoder(patch_num, d_model, nhead))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        # b,c,l
        # print(x.shape)
        # print(x.shape)
        # x=self.L(x)
        # print(x.shape)
        # x=self.T(x)
        # print(x.shape)
        return self.net(x)


class PatchDecoder(nn.Module):

    def __init__(
        self,
        in_len: int,
        hid_len: int,
        in_chn: int,
        hid_chn: int,
        out_chn,
        patch_size: int,
        hid_pch: int,
        d_model: int,
        patch_num: int,
        nhead: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        # self.net = nn.Sequential()
        self.net=[]
        inter_patch_mlp = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ,
                         drop)
        channel_wise_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)
        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(1, patch_size)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        # self.net.append(Rearrange("b c l1 -> b c l1 1"))
        # self.net.append(linear)
        # self.net.append(norm_class(in_chn))
        # self.net.append(intra_patch_mlp)
        # self.net.append(norm_class(in_chn))
        # self.net.append(inter_patch_mlp)
        # self.net.append(norm_class(in_chn))
        # self.net.append(channel_wise_mlp)
        self.net.append(TSTEncoder(patch_num,d_model,nhead))
        self.net.append(Rearrange("(b c) l1 l2 -> b c l1 l2",c=in_chn))
        self.net.append(nn.Linear( d_model,patch_size))
        self.net.append(Rearrange("b c l1 l2 -> b c (l1 l2)"))
        self.net=nn.Sequential(*self.net)
    def forward(self, x):
        # b,c,l
        return self.net(x)


class PredictionHead(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 hid_len,
                 in_chn,
                 out_chn,
                 hid_chn,
                 activ,
                 drop=0.0) -> None:
        super().__init__()
        # self.net = nn.Sequential()
        self.net=[]
        if in_chn != out_chn:
            c_jump_conn = "proj"
        else:
            c_jump_conn = "trunc"
        self.net.append(
            MLPBlock(1,
                in_chn,
                hid_chn,
                out_chn,
                activ=activ,
                drop=drop,
                jump_conn=c_jump_conn))
        self.net.append(
            MLPBlock(2,
                in_len,
                hid_len,
                out_len,
                activ=activ,
                drop=drop,
                jump_conn='proj'))
        self.net=nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class MSDMixer(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 in_chn,
                 ex_chn,
                 out_chn,
                 patch_sizes,
                 hid_len,
                 hid_chn,
                 hid_pch,
                 hid_pred,
                 norm,
                 last_norm,
                 activ,
                 drop,
                 dmodel,
                 nhead,
                 revin,
                 affine,
                 reduction="sum") -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.reduction = reduction
        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.patch_sizes = patch_sizes
        self.paddings = []
        self.revin=revin
        # all_chn = in_chn + ex_chn
        if self.revin: self.revin_layer = RevIN(in_chn, affine=affine, subtract_last=False)

        for i, patch_size in enumerate(patch_sizes):
            res = in_len % patch_size
            padding = (patch_size - res) % patch_size
            self.paddings.append(padding)
            padded_len = in_len + padding
            patch_num=padded_len//patch_size
            # print(patch_num,dmodel,nhead)
            self.patch_encoders.append(PatchEncoder(padded_len, hid_len, in_chn, hid_chn,
                          in_chn, patch_size, hid_pch,dmodel,patch_num,nhead, norm, activ, drop))
            self.patch_decoders.append(PatchDecoder(padded_len, hid_len, in_chn, hid_chn, in_chn,
                        patch_size, hid_pch,dmodel,patch_num,nhead, norm, activ, drop))
            if out_len != 0 and out_chn != 0:
                self.pred_heads.append(Flatten_Head(False,self.in_chn,dmodel*patch_num,self.out_len))
            else:
                self.pred_heads.append(nn.Identity())

    def forward(self, x, x_mark=None, x_mask=None):
        x = rearrange(x, "b l c -> b c l")
        if x_mark is not None:
            x_mark = rearrange(x_mark, "b l c -> b c l")
        if x_mask is not None:
            x_mask = rearrange(x_mask, "b l c -> b c l")


        # if self.last_norm:
        #     x_last = x[:, :, [-1]].detach()
        #     x = x - x_last
        #     if x_mark is not None:
        #         x_mark_last = x_mark[:, :, [-1]].detach()
        #         x_mark = x_mark - x_mark_last
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0, 2, 1)


        y_pred = []
        for i in range(len(self.patch_sizes)):
            x_in = x
            if x_mark is not None:
                x_in = torch.cat((x, x_mark), 1)
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)

            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            pred = self.pred_heads[i](emb)
            if x_mask is not None:
                comp = comp * x_mask
            x = x - comp

            if self.out_len != 0 and self.out_chn != 0:
                y_pred.append(pred)

        if self.out_len != 0 and self.out_chn != 0:
            y_pred = reduce(torch.stack(y_pred, 0), "h b c l -> b c l",
                            self.reduction)
            # if self.last_norm and self.out_chn == self.in_chn:
            #     y_pred += x_last
            if self.revin:
                y_pred = y_pred.permute(0, 2, 1)
                y_pred = self.revin_layer(y_pred, 'denorm')
                y_pred = y_pred.permute(0, 2, 1)
            y_pred = rearrange(y_pred, "b c l -> b l c")
            return y_pred, x
        else:
            return None, x



class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len=configs.pred_len


        self.channel_in=configs.enc_in
        self.channel_out=configs.c_out

        self.ex_chn=0
        self.drop = 0.5
        self.patch_sizes = [48,24, 4]
        self.hid_chn = 512
        self.hid_len = 512
        self.hid_pred = 512
        self.hid_pch = 512

        self.norm = None
        self.last_norm = True

        self.activ = "gelu"
        self.drop = 0.5

        self.dmodel=configs.d_model
        self.nhead=configs.n_heads
        self.revin=True
        self.affine=True

        self.MSD=MSDMixer(self.seq_len,self.pred_len,self.channel_in,self.ex_chn,self.channel_out,self.patch_sizes,
                          self.hid_len,self.hid_chn,self.hid_pch,self.hid_pred,self.norm,self.last_norm,self.activ,
                          self.drop,self.dmodel,self.nhead,self.revin,self.affine)
        self.DML=DML(configs)
    def forward(self ,x,batch_x_mark, dec_inp, batch_y_mark, batch_y=None):

        res = self.DML(x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
        y_pred, _=self.MSD(x)


        return y_pred+res