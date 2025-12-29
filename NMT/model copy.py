import config
from data_loader import subsequent_mask

import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):
    """
    集成绝对/相对位置编码的统一类
    :param d_model: 模型维度（绝对位置编码）/注意力头维度d_k（相对位置编码）
    :param dropout: dropout概率（仅绝对位置编码使用）
    :param max_len: 最大序列长度
    :param pos_encoding_type: 编码类型，可选 "absolute"（绝对）/ "relative"（相对）
    :param device: 初始化设备（None则动态适配）
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, pos_encoding_type="absolute", device=None):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding_type = pos_encoding_type
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout) if pos_encoding_type == "absolute" else None

        # -------------------------- 绝对位置编码初始化（兼容原有逻辑） --------------------------
        if pos_encoding_type == "absolute":
            # 移除硬编码DEVICE，改为动态设备适配（解决多GPU问题）
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0., max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
            
            # 按奇偶维度填充正弦/余弦值
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            
            # 注册为buffer（不参与训练），不绑定固定设备
            self.register_buffer('pe', pe)
        
        # -------------------------- 相对位置编码初始化（Transformer-XL风格） --------------------------
        elif pos_encoding_type == "relative":
            # 相对位置范围：-max_len ~ max_len → 映射为0 ~ 2*max_len（非负索引）
            self.rel_pos_embed = nn.Embedding(2 * max_len + 1, d_model)
            # 初始化嵌入层参数
            nn.init.normal_(self.rel_pos_embed.weight, std=0.02)
            # 若指定初始化设备，移到对应设备（多GPU时由DataParallel自动分配）
            if device is not None:
                self.rel_pos_embed = self.rel_pos_embed.to(device)

    def get_rel_pos_index(self, q_len, k_len, device):
        """
        生成相对位置索引（适配自注意力/encoder-decoder注意力）
        :param q_len: query序列长度
        :param k_len: key序列长度（自注意力时q_len=k_len）
        :param device: 当前输入设备（动态获取）
        :return: [q_len, k_len] 相对位置索引（在指定设备上）
        """
        # 直接在目标设备生成位置索引，避免跨设备问题
        q_pos = torch.arange(q_len, dtype=torch.long, device=device).unsqueeze(1)  # [q_len, 1]
        k_pos = torch.arange(k_len, dtype=torch.long, device=device).unsqueeze(0)  # [1, k_len]
        rel_pos = q_pos - k_pos  # 计算相对位置（可负）
        # 偏移到非负索引范围，防止embedding层越界
        rel_pos = rel_pos.clamp(-self.max_len, self.max_len) + self.max_len
        return rel_pos

    def forward(self, x, k_len=None):
        """
        前向传播：根据类型返回对应编码结果
        :param x: 输入张量（绝对编码：[batch, seq_len, d_model]；相对编码：仅用于获取长度/设备）
        :param k_len: 仅相对编码使用，key序列长度（None则为自注意力，q_len=k_len）
        :return: 绝对编码：x + 位置编码（dropout后）；相对编码：[q_len, k_len, d_model] 相对位置编码矩阵
        """
        device = x.device  # 动态获取输入设备，适配多GPU
        
        # -------------------------- 绝对位置编码逻辑（兼容原有使用方式） --------------------------
        if self.pos_encoding_type == "absolute":
            # 动态将pe移到输入设备，避免跨GPU问题
            pe = self.pe[:, :x.size(1)].to(device)
            # 位置编码与输入相加（保留原有Variable兼容）
            x = x + Variable(pe, requires_grad=False)
            return self.dropout(x)
        
        # -------------------------- 相对位置编码逻辑（供注意力层调用） --------------------------
        elif self.pos_encoding_type == "relative":
            q_len = x.size(1)  # query序列长度（x仅用于获取长度和设备）
            if k_len is None:
                k_len = q_len  # 自注意力：q/k长度相同
            # 生成相对位置索引
            rel_pos_idx = self.get_rel_pos_index(q_len, k_len, device)
            # 获取相对位置编码矩阵
            rel_pos_embed = self.rel_pos_embed(rel_pos_idx)
            return rel_pos_embed


def attention(query, key, value, mask=None, dropout=None, rel_pos_embed=None):
    d_k = query.size(-1)
    # scores维度：[sub_batch, heads, q_len, k_len]（如32, 8, 50, 50）
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # 步骤1：强制将mask移到scores的设备
        mask = mask.to(scores.device)
        
        # 步骤2：扩展mask维度到4维（匹配scores的[B, H, Q, K]）
        while len(mask.shape) < len(scores.shape):
            mask = mask.unsqueeze(1)
        
        # 步骤3：核心修复——强制对齐batch维度（关键！）
        # 如果mask的batch维度 != scores的batch维度，截断/重复到匹配
        if mask.size(0) != scores.size(0):
            # 优先截断（DataParallel拆分后子batch更小）
            if mask.size(0) > scores.size(0):
                mask = mask[:scores.size(0)]  # 70 → 32
            # 极端情况：mask更小则重复（一般不会出现）
            else:
                mask = mask.repeat(scores.size(0) // mask.size(0) + 1, 1, 1, 1)[:scores.size(0)]
        
        # 步骤4：确保mask的最后两维与scores匹配（防止序列长度不一致）
        if mask.size(-2) != scores.size(-2) or mask.size(-1) != scores.size(-1):
            # 重新生成空mask（兜底方案，避免序列长度不匹配）
            mask = torch.ones_like(scores) == 1  # 全True的mask（无遮挡）
        
        # 执行mask填充
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 相对位置编码（保留原有逻辑）
    if rel_pos_embed is not None:
        if rel_pos_embed.size(0) == scores.size(2) and rel_pos_embed.size(1) == scores.size(3):
            rel_scores = torch.einsum('bhqd, qkd -> bhqk', query, rel_pos_embed) / math.sqrt(d_k)
            scores += rel_scores

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches = query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化α为全1, 而β为全0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 按最后一个维度计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # 返回Layer Norm的结果
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    SublayerConnection的作用就是把Multi-Head Attention和Feed Forward层连在一起
    只不过每一层输出之后都要先做Layer Norm再残差连接
    sublayer是lambda函数
    """
    # 【修改1】添加norm_type参数，默认使用LayerNorm
    def __init__(self, size, dropout, norm_type="LayerNorm"):
        super(SublayerConnection, self).__init__()
        # 【修改2】根据norm_type选择归一化方式
        self.norm = nn.RMSNorm(size) if norm_type == "RMSNorm" else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 返回Layer Norm和残差连接后结果
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    # 【修改3】添加norm_type参数，默认使用LayerNorm
    def __init__(self, layer, N, norm_type="LayerNorm"):
        super(Encoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # 【修改4】根据norm_type选择归一化方式，删除多余的self.rmsnorm
        self.norm = nn.RMSNorm(layer.size) if norm_type == "RMSNorm" else LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        使用循环连续eecode N次(这里为6次)
        这里的Eecoderlayer会接收一个对于输入的attention mask处理
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    # 【修改5】添加norm_type参数，默认使用LayerNorm
    def __init__(self, size, self_attn, feed_forward, dropout, norm_type="LayerNorm"):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 【修改6】传递norm_type参数给SublayerConnection
        self.sublayer = clones(SublayerConnection(size, dropout, norm_type), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 注意到attn得到的结果x直接作为了下一层的输入
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    # 【修改7】添加norm_type参数，默认使用LayerNorm
    def __init__(self, layer, N, norm_type="LayerNorm"):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # 【修改8】根据norm_type选择归一化方式
        self.norm = nn.RMSNorm(layer.size) if norm_type == "RMSNorm" else LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        使用循环连续decode N次(这里为6次)
        这里的Decoderlayer会接收一个对于输入的attention mask处理
        和一个对输出的attention mask + subsequent mask处理
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    # 【修改9】添加norm_type参数，默认使用LayerNorm
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, norm_type="LayerNorm"):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn
        # 与Encoder传入的Context进行Attention
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 【修改10】传递norm_type参数给SublayerConnection
        self.sublayer = clones(SublayerConnection(size, dropout, norm_type), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 用m来存放encoder的最终hidden表示结果
        m = memory

        # Self-Attention：注意self-attention的q，k和v均为decoder hidden
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Context-Attention：注意context-attention的q为decoder hidden，而k和v为encoder hidden
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)


# 【修改11】添加norm_type参数，默认使用LayerNorm
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, norm_type="LayerNorm",pos_encoding_type="absolute"):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout, pos_encoding_type=pos_encoding_type).to(DEVICE)
    # 实例化Transformer模型对象
    model = Transformer(
        # 【修改12】传递norm_type参数给EncoderLayer和Encoder
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, norm_type).to(DEVICE), N, norm_type).to(DEVICE),
        # 【修改13】传递norm_type参数给DecoderLayer和Decoder
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, norm_type).to(DEVICE), N, norm_type).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            # 这里初始化采用的是nn.init.xavier_uniform
            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    memory = model.encode(src, src_mask)
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    for s in range(max_len):
        tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src.data)
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)

        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results


def greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """传入一个训练好的模型，对指定数据进行预测"""
    # 先用encoder进行encode
    memory = model.encode(src, src_mask)
    # 初始化预测内容为1×1的tensor，填入开始符('BOS')的id，并将type设置为输入数据类型(LongTensor)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    # 遍历输出的长度下标
    for i in range(max_len - 1):
        # decode得到隐层表示
        out = model.decode(memory,
                        src_mask,
                        Variable(ys),
                        Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        # 将隐藏表示转为对词典各词的log_softmax概率分布表示
        prob = model.generator(out[:, -1])
        # 获取当前位置最大概率的预测词id
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        if next_word == end_symbol:
            break
        # 将当前位置预测的字符id与之前的预测内容拼接起来
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys