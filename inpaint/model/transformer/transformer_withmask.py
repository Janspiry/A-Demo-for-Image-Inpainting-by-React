import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播.
        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention * (1.0-attn_mask.float())
            # attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        if attn_mask is not None:
            attention = attention * (1.0-attn_mask.float())
            attention.clamp(min=1e-8)

        # if attn_mask is not None:
        #     # 给需要mask的地方设置一个负无穷
        #     # attention = attention * attn_mask
        #     attention = attention.masked_fill_(attn_mask, torch.tensor(-np.inf).cuda())
        # attention = self.softmax(attention)
        # if attn_mask is not None:
        #     # attention = attention * (1.0-attn_mask.float())
        #     attention.clamp(min=1e-8)
        #     attention  = torch.nan_to_num(attention)
        # 计算softmax
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        # print('v.sum()',v.sum())
        context = torch.bmm(attention, v)
        # print('context.sum()', context.sum(), 'attention.sum()', attention.sum())
        
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


def padding_mask(seq_k, seq_q):
    # seq_k 和 seq_q 的形状都是 [B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len, _ = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, int(len) + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)



class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, masks=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, masks)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""

    def __init__(self,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

    def forward(self, inputs, masks=None):
        output = inputs
        # self_attention_mask = padding_mask(inputs, inputs)
        
        
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, masks=masks)
            attentions.append(attention)

        return output, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, masks=None):
        # self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(
          dec_inputs, dec_inputs, dec_inputs, masks)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(
          enc_outputs, enc_outputs, dec_output, masks)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):

    def __init__(self,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])
        
    def forward(self, inputs, enc_output, masks=None):
        self_attentions = []
        context_attentions = []
        output = inputs
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, masks=masks)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions




class Transformer(nn.Module):
    def __init__(self,
               N,
               dim,
               model_dim=512,
               num_layers=6,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.2,
               with_attn=False,
               condition=False
               ):
        super(Transformer, self).__init__()
        self.N = N
        self.dim = dim
        self.with_attn = with_attn
        self.condition = condition
        self.patch_to_embedding = nn.Linear(dim, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, N, model_dim)) # 1, N, dim
        if self.condition:
            # dim+dim : 上采样通道扩增
            self.condition_to_embedding = nn.Linear(dim, model_dim)
            self.condition_pos_embedding = nn.Parameter(torch.randn(1, N, model_dim)) # 1, N, dim
        self.encoder = Encoder(num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(num_layers, model_dim, num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, dim, bias=False)
    def forward(self, x, condition=None, masks=None):
        b, N, C = x.size()
        # print('x.size()',x.size())
        # condition的利用方式存疑？
        output = None
        if condition is None:
            x = self.patch_to_embedding(x) # b, N, dim
            x += self.pos_embedding
            x, enc_self_attn = self.encoder(x, masks=masks)
            output, dec_self_attn, ctx_attn = self.decoder(inputs=x, enc_output=x, masks=masks)
        elif self.condition:
            '''考虑condition'''
            x = self.patch_to_embedding(x) # b, N, dim
            x += self.pos_embedding
            condition = self.condition_to_embedding(condition)
            condition += self.condition_pos_embedding
            condition, enc_self_attn = self.encoder(condition, masks=masks)
            output, dec_self_attn, ctx_attn = self.decoder(inputs=x, enc_output=condition, masks=masks)
        output = self.linear(output)        
        if self.with_attn:
            return output, enc_self_attn, dec_self_attn, ctx_attn
        else:
            return output

class TransformerLayer(nn.Module):
    def __init__(self, size, patch_size, condition=False, MiniTransFormer=None, with_mask=False):
        super(TransformerLayer, self).__init__()
        # size: [c, h, w]
        self.input_dim, self.H, self.W = size
        self.p = patch_size # patch_size
        self.patch_dim = self.input_dim*(self.p*self.p)
        self.patch_num = (self.H//self.p)*(self.W//self.p)

        model_dim = 256
        num_layers = 6
        num_heads = 8
        ffn_dim = 512
        if MiniTransFormer is not None:
            model_dim, num_layers, num_heads, ffn_dim = MiniTransFormer
        self.condition = condition
        self.transformer_h = Transformer(
            N=self.patch_num,
            dim=self.patch_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            condition=self.condition
        )

        self.mask_patch = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0)


    def forward(self, x, condition=None, masks=None):
        b, c, h, w = x.size()
        p = self.p
        new_h, new_w = h//p, w//p
        if masks is not None:
            # masks = masks.repeat(1, c, 1, 1)    # mask[b, N, 1] -> [b, N, N]
            masks = F.interpolate(masks, size=x.size()[2:4])
            masks = self.mask_patch(masks).permute(0,2,1).contiguous() # b, N//p*2, c
            masks = masks.mean([2])
            masks = masks.gt(0.).view(b, -1, 1) # (b, N//P*2, 1) # set 1 where need mask
            # print('batch: {}/{}'.format(masks.sum()/masks.size()[0], masks.size()[1]))
            # print('single: {}/{}'.format(masks[0].sum(), masks[0].size()[0]))
            # tmp = masks[0]
            # for i in tmp:
            #     print(i, end=' ')
            masks = masks.repeat(1, 1, new_h*new_w)    # mask[b, N, 1] -> [b, N, N]
        # P = 8, patch_size 大小， 32 * 32 
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) # b, N, p^2*C -> b, 1024, 320

        if condition is not None:
            condition = rearrange(condition, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) 

        x = self.transformer_h(x, condition, masks)
        x = rearrange(x, ' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = new_h, w = new_w, p1 = p, p2 = p) # b, c, h, w

        return x

if __name__ == "__main__":
    x = torch.randn(4, 3, 16, 16)
    mask = torch.randn(4, 1, 16, 16)
    transformer = TransformerLayer(
        size=[3, 16, 16],patch_size=1, MiniTransFormer=[128, 1, 4, 512]
    )
    transformer(x, masks=mask)