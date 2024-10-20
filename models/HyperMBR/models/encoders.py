"""Graph encoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class AdaHypBR(Encoder):
    """
    Hyperbolic-Bundle-GCN.
    """

    def __init__(self, c, info,args,levelflag):
        super(AdaHypBR, self).__init__(c)
        self.manifold = getattr(manifolds, info.manifold)()
        assert info.num_layers >= 1
        assert levelflag in ["itemLevel","bundleLevel"]
        if levelflag=="itemLevel":
            dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(info,args,args.itemLevel_c)
            self.curvatures.append(self.c)
        elif levelflag=="bundleLevel":
            dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(info, args, args.bundleLevel_c)
            self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicBundleGraphConvolution(
                    self.manifold, in_dim, out_dim, c_in, c_out, info.dropout, act, args.bias, args.use_att, args.local_agg,args.alpha, args.n_heads,args.concat,args.device
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(AdaHypBR, self).encode(x_hyp, adj)