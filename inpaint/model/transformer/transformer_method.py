from .transformer import TransformerLayer
from .transformer_withmask import TransformerLayer as Mask_TransformerLayer
from .swin_transformer import TransformerModule as Swin_TransformerLayer
import torch
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self, size, patch_size, MiniTransFormer=None, use_local=False, use_global=False):
        super(TransformerBlock, self).__init__()
        model_dim = 256
        num_layers = 6
        num_heads = 8
        ffn_dim = 512
        self.p = patch_size # patch_size
        if MiniTransFormer is not None:
            model_dim, num_layers, num_heads, ffn_dim = MiniTransFormer

        self.transformer_global = TransformerLayer(
            size=size,
            patch_size=patch_size, 
            MiniTransFormer=MiniTransFormer
        )
        self.transformer_local = Swin_TransformerLayer(
            in_channels=size[0],
            hidden_dimension=model_dim,
            layers=2,
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=8,
            relative_pos_embedding=True
        )
        self.use_local = use_local
        self.use_global = use_global
        assert self.use_local or self.use_global, 'self.use_local and self.use_global are false.'
        
    def forward(self, x):
        b, c, h, w = x.size()
        if(self.use_global):
            x = self.transformer_global(x)
        if(self.use_local):
            x = self.transformer_local(x)
        
        return x

class MaskTransformer(nn.Module):
    def __init__(self, size, patch_size, MiniTransFormer=None, use_local=False, use_global=False):
        super(MaskTransformer, self).__init__()
        model_dim = 256
        num_layers = 6
        num_heads = 8
        ffn_dim = 512
        self.p = patch_size # patch_size
        if MiniTransFormer is not None:
            model_dim, num_layers, num_heads, ffn_dim = MiniTransFormer

        self.transformer_global = Mask_TransformerLayer(
            size=size,
            patch_size=patch_size, 
            MiniTransFormer=MiniTransFormer
        )
        self.transformer_local = Swin_TransformerLayer(
            in_channels=size[0],
            hidden_dimension=model_dim,
            layers=2,
            patch_size=patch_size,
            num_heads=num_heads,
            window_size=8,
            relative_pos_embedding=True
        )
        self.use_local = use_local
        self.use_global = use_global
        assert self.use_local or self.use_global, 'self.use_local and self.use_global are false.'
        
    def forward(self, x, masks):
        b, c, h, w = x.size()
        if(self.use_global):
            x = self.transformer_global(x, masks=masks)
        if(self.use_local):
            x = self.transformer_local(x)
        
        return x