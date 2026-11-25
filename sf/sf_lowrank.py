"""Low-rank CTPA compression candidates; existing SVD method, no accuracy guarantee."""
from copy import deepcopy
import torch
from torch import nn


def factors(weight,rank):
    weight=weight.detach().float().cpu()
    u,s,vh=torch.linalg.svd(weight,full_matrices=False)
    assert 0<rank<=len(s)
    root=s[:rank].sqrt()
    left=u[:,:rank]*root[None,:]
    right=root[:,None]*vh[:rank]
    energy=float(s[:rank].square().sum()/s.square().sum())
    return left,right,energy


class FactorizedLinear(nn.Module):
    def __init__(self,in_features,out_features,rank,bias=True):
        super().__init__()
        self.in_features,self.out_features,self.rank=in_features,out_features,rank
        self.first=nn.Linear(in_features,rank,bias=False)
        self.second=nn.Linear(rank,out_features,bias=bias)

    @classmethod
    def from_weight(cls,weight,bias,rank):
        obj=cls(weight.shape[1],weight.shape[0],rank,bias is not None)
        left,right,energy=factors(weight,rank)
        with torch.no_grad():
            obj.first.weight.copy_(right);obj.second.weight.copy_(left)
            if bias is not None:obj.second.bias.copy_(bias.detach().cpu())
        obj.retained_svd_energy=energy
        return obj

    def forward(self,x):return self.second(self.first(x))


class FactorizedPointwise(nn.Module):
    def __init__(self,in_channels,out_channels,rank,bias=False):
        super().__init__()
        self.first=nn.Conv2d(in_channels,rank,1,bias=False)
        self.second=nn.Conv2d(rank,out_channels,1,bias=bias)

    @classmethod
    def from_conv(cls,conv,rank):
        assert conv.kernel_size==(1,1) and conv.stride==(1,1) and conv.groups==1
        obj=cls(conv.in_channels,conv.out_channels,rank,conv.bias is not None)
        left,right,energy=factors(conv.weight[:,:,0,0],rank)
        with torch.no_grad():
            obj.first.weight.copy_(right[:,:,None,None]);obj.second.weight.copy_(left[:,:,None,None])
            if conv.bias is not None:obj.second.bias.copy_(conv.bias.detach().cpu())
        obj.retained_svd_energy=energy
        return obj

    def forward(self,x):return self.second(self.first(x))


class LowRankSelfAttention(nn.Module):
    """Full head/output dimensions, with a joint low-rank input projection."""
    def __init__(self,source,rank):
        super().__init__()
        assert source.batch_first and source.dropout==0 and source._qkv_same_embed_dim
        assert source.bias_k is None and source.bias_v is None and not source.add_zero_attn
        self.embed_dim=source.embed_dim;self.num_heads=source.num_heads;self.batch_first=True
        self.qkv=FactorizedLinear.from_weight(source.in_proj_weight,source.in_proj_bias,rank)
        self.out_proj=deepcopy(source.out_proj).float().cpu()

    def forward(self,query,key,value,need_weights=True,**kwargs):
        assert query is key and key is value,'Self-attention only'
        assert not kwargs.get('is_causal',False) and kwargs.get('attn_mask') is None and kwargs.get('key_padding_mask') is None
        b,n,c=query.shape;d=c//self.num_heads
        q,k,v=self.qkv(query).reshape(b,n,3,self.num_heads,d).permute(2,0,3,1,4).unbind(0)
        attention=((q@k.transpose(-2,-1))*(d**-.5)).softmax(-1)
        out=(attention@v).transpose(1,2).reshape(b,n,c)
        return self.out_proj(out),attention.mean(1) if need_weights else None


VARIANTS={
 'sf_lr_balanced':dict(qkv_rank=64,ffn_rank=32,fusion_rank=96),
 'sf_lr_attention_full':dict(qkv_rank=None,ffn_rank=32,fusion_rank=64),
 'sf_lr_aggressive':dict(qkv_rank=48,ffn_rank=32,fusion_rank=80),
}


def compress_detector(detector,variant):
    """Copy, then replace only the CTPA block's selected linear maps."""
    model=deepcopy(detector).float().cpu()
    blocks=[m for m in model.modules() if type(m).__name__=='CNNTransformerParallel']
    assert len(blocks)==1
    block=blocks[0];cfg=VARIANTS[variant]
    if cfg['qkv_rank'] is not None:block.attn=LowRankSelfAttention(block.attn,cfg['qkv_rank'])
    for index in (0,2):
        old=block.ffn[index]
        block.ffn[index]=FactorizedLinear.from_weight(old.weight,old.bias,cfg['ffn_rank'])
    block.fuse[0]=FactorizedPointwise.from_conv(block.fuse[0],cfg['fusion_rank'])
    block.np=sum(p.numel() for p in block.parameters())
    model.compression_variant=variant
    return model
