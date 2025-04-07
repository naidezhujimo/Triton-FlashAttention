import torch
import triton
import triton.language as tl
from timeit import Timer

@triton.jit
def flash_attention_v1(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 新增：二维网格布局（序列分块，特征分块）
    pid_m = tl.program_id(0)  # 处理序列维度的分块
    pid_d = tl.program_id(1)  # 新增：处理特征维度的分块
    
    # 计算分块起始位置
    start_m = pid_m * BLOCK_M
    start_d = pid_d * BLOCK_D  # 新增：特征维度分块起始
    
    # 生成局部偏移
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = start_d + tl.arange(0, BLOCK_D)  # 修改：基于分块起始的特征偏移

    # 初始化累加器（调整为分块大小）
    m_prev = tl.full((BLOCK_M, ), float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)  # 修改：BLOCK_D维度

    # 加载Q分块 [BLOCK_M, BLOCK_D]
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model),
        other=0.0
    ).to(tl.float32)
    
    # 遍历K/V块
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        # 加载K分块 [BLOCK_N, BLOCK_D]
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_km + offs_d[None, :],  # 特征维度对齐
            mask=mask_n[:, None] & (offs_d[None, :] < d_model),
            other=0.0
        ).to(tl.float32)
        
        # V分块加载需对齐特征维度
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :],  # 增加特征偏移
            mask=mask_n[:, None] & (offs_d[None, :] < d_model),
            other=0.0
        ).to(tl.float32)
        
        # 计算QK^T分块 [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k.T)
        s *= 1.0 / tl.sqrt(tl.cast(d_model, tl.float32))

        # 掩码无效位置
        s = tl.where(mask_n[None, :], s, float('-inf'))

        # 在线Softmax
        m_curr = tl.maximum(tl.max(s, axis=1), m_prev)
        alpha = tl.exp(m_prev - m_curr)
        beta = tl.exp(s - m_curr[:, None])
        l_curr = alpha * l_prev + tl.sum(beta, axis=1)
        p = beta / l_curr[:, None]
        
        # 分块累加需限制维度
        acc += tl.dot(p, v)  # [BLOCK_M, BLOCK_D]

        # 保存中间变量
        m_prev = m_curr
        l_prev = l_curr

    # 写入最终结果（对齐特征分块）
    tl.store(
        o_ptr + offs_m[:, None] * stride_qm + offs_d[None, :],
        acc,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model)
    )

def call_flash_attention_v1(q, k, v):
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    seq_len, d_model = q.shape
    o = torch.empty_like(q)

    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 64
    
    # 修改为二维网格布局
    grid = (
        triton.cdiv(seq_len, BLOCK_M),  # 序列维度分块数
        triton.cdiv(d_model, BLOCK_D)   # 新增：特征维度分块数
    )

    flash_attention_v1[grid](
        q, k, v, o,
        seq_len, d_model,
        q.stride(0), k.stride(0), v.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o



@triton.jit
def flash_attention_v2(
    # 张量指针
    q_ptr, k_ptr, v_ptr, o_ptr,
    # 元数据
    seq_len, head_dim: tl.constexpr,
    # 内存步幅
    q_stride_m, q_stride_h, # Q的步幅 [seq_len, num_heads, head_dim]
    k_stride_m, k_stride_h, # K的步幅
    v_stride_m, v_stride_h, # V的步幅
    o_stride_m, o_stride_h, # O的步幅
    # 超参数
    BLOCK_M: tl.constexpr, # Q块大小
    BLOCK_N: tl.constexpr, # K/V块大小
    NUM_HEADS: tl.constexpr, # 头数
    IS_CAUSAL: tl.constexpr # 是否因果掩码
):
    # 1. 计算程序ID与初始偏移量
    pid_head = tl.program_id(0) # 头索引
    pid_m = tl.program_id(1) # 每个头内处理Q块的索引

    # 当前Q块的起始位置
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 2. 初始化累加器
    m_i = tl.full((BLOCK_M, ), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)

    # 3. 加载Q块(使用向量化加载)
    q_offset = pid_head * q_stride_h + offs_m[:, None] * q_stride_m
    q = tl.load(q_ptr + q_offset + tl.arange(0, head_dim)[None, :] * q_stride_h,
                mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
                other=0.0).to(tl.float32)
    
    # 4. 主循环处理K/V块
    for start_n in range(0, (seq_len + BLOCK_M - 1) // BLOCK_N * BLOCK_N, BLOCK_N):
        # 4.1 计算当前K/V块的有限范围
        valid_n = start_n + offs_n < seq_len
        # 4.2 加载K块
        k_offset = pid_head * k_stride_h + (start_n + offs_n)[:, None] * k_stride_m
        k = tl.load(k_ptr + k_offset + tl.arange(0, head_dim)[None, :] * 1,
                    mask=valid_n[:, None] & (tl.arange(0, head_dim)[None, :] < head_dim), 
                    other=0.0).to(tl.float32)
        # 4.3 加载V块
        v_offset = pid_head * v_stride_h + (start_n + offs_n)[:, None] * v_stride_m
        v = tl.load(v_ptr + v_offset + tl.arange(0, head_dim)[None, :] * 1,
                    mask=valid_n[:, None] & (tl.arange(0, head_dim)[None, :] < head_dim),
                    other=0.0).to(tl.float32)
        
        # 4.4 计算QK^T(启用Tensor Core加速)
        s = tl.dot(q, k.T.to(q.dtype))
        s = s * (1.0 / tl.sqrt(tl.cast(head_dim, tl.float32)))

        # 4.5 处理因果掩码
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None]) >= (start_n + offs_n[None, :])
            s = tl.where(causal_mask, s, float('-inf'))

        # 4.6 在线Softmax更新
        # 计算当前块的行最大值
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        # 计算指数和
        alpha = tl.exp(m_i - m_curr) # 旧最大衰减因子
        beta = tl.exp(s - m_curr[:, None]) # 当前块指数
        l_curr = alpha * l_i + tl.sum(beta, axis=1)
        # 更新累加器
        p = beta / l_curr[:, None]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        # 4.7 保存中间变量
        m_i = m_curr
        l_i = l_curr

    # 最终归一化并存储结果
    o = acc / l_i[:, None]
    # 存储到全局变量
    o_offset = pid_head * o_stride_h + offs_m[:, None] * o_stride_m
    tl.store(o_ptr + o_offset + tl.arange(0, head_dim)[None, :] * 1,
             o.to(o_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim))

def call_flash_attention_v2(q, k, v, is_causal=False):
    assert q.dim() == 3, "Input should be [seq_len, num_heads, head_dim]"
    seq_len, num_heads, head_dim = q.shape
    o = torch.empty_like(q)
    
    config = {
        'BLOCK_M': 128,
        'BLOCK_N': 64,
        'num_warps': 8,
        'num_stages': 3,
    }
    
    # 网格维度：每个头独立计算，每个头内划分Q块
    grid = (num_heads, triton.cdiv(seq_len, config['BLOCK_M']))
    
    flash_attention_v2[grid](
        q, k, v, o,
        seq_len, head_dim,
        # 内存步幅计算（假设输入为连续张量）
        q.stride(1), q.stride(0),
        k.stride(1), k.stride(0),
        v.stride(1), v.stride(0),
        o.stride(1), o.stride(0),
        NUM_HEADS=num_heads,
        IS_CAUSAL=is_causal,
        **config
    )
    return o


@triton.jit
def flash_attention_v3(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, head_dim: tl.constexpr,
    stride_qm, stride_qh,
    stride_km, stride_kh,
    stride_vm, stride_vh,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr
):
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 初始化累加器（根据量化模式调整精度）
    acc_dtype = tl.float8e5 if USE_FP8 else tl.float32
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=acc_dtype)

    # 加载Q块（直接存储为FP8）
    if USE_FP8:
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :] * stride_qh,
            mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
            other=0.0
        ).to(tl.float8e5)
        q_scale = 127.0 / tl.max(tl.abs(q.to(tl.float32))) + 1e-6
    else:
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :] * stride_qh,
            mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
            other=0.0
        ).to(tl.float32)  # 原始逻辑

    # 初始化K/V块指针（支持FP8加载）
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_km, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_vm, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(0, 1)
    )

    # 主循环
    for start_n in range(0, seq_len, BLOCK_N):
        # 加载当前K/V块（关键修改：直接存储为FP8）
        curr_k = tl.load(k_block_ptr, boundary_check=(0,)).to(tl.float8e5 if USE_FP8 else tl.float32)
        curr_v = tl.load(v_block_ptr, boundary_check=(0,)).to(tl.float8e5 if USE_FP8 else tl.float32)

        if USE_FP8:
            # FP8计算（无需反量化）
            k_scale = 127.0 / tl.max(tl.abs(curr_k.to(tl.float32))) + 1e-6
            v_scale = 127.0 / tl.max(tl.abs(curr_v.to(tl.float32))) + 1e-6
            
            # 计算QK^T（使用FP8 Tensor Core）
            s = tl.dot(q, tl.trans(curr_k), allow_tf32=True).to(tl.float32)
            s = s * (1.0 / (q_scale * k_scale))  # 缩放因子融合
        else:
            # 原始FP32计算
            s = tl.dot(q.to(tl.float32), tl.trans(curr_k.to(tl.float32)))
        
        s = s * (1.0 / tl.sqrt(tl.cast(head_dim, tl.float32)))

        # 在线Softmax
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        alpha = tl.exp(m_i - m_curr)
        beta = tl.exp(s - m_curr[:, None])
        l_curr = alpha * l_i + tl.sum(beta, axis=1)
        p = beta / (l_curr[:, None] + 1e-6)

        # 累加输出（FP8直接累加）
        if USE_FP8:
            p = p.to(tl.float8e5)
            curr_v_scaled = (curr_v * v_scale).to(tl.float8e5)
            acc = tl.dot(p, curr_v_scaled).to(acc_dtype) + acc
        else:
            acc += tl.dot(p, curr_v.to(tl.float32))

        # 更新中间变量
        m_i = m_curr
        l_i = l_curr

        # 预加载下一个块
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    # 存储结果（保持FP8格式）
    tl.store(
        o_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :],
        acc.to(tl.float8e5 if USE_FP8 else tl.float32),
        mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim)
    )



def call_flash_attention_v3(q, k, v, use_fp8=False):
    assert q.dim() == 3, "Input should be [seq_len, num_heads, head_dim]"
    
    # 强制输入输出为FP8格式（PyTorch 2.1+）
    if use_fp8:
        q = q.to(torch.float8_e5m2)
        k = k.to(torch.float8_e5m2)
        v = v.to(torch.float8_e5m2)
        o = torch.empty_like(q, dtype=torch.float8_e5m2)
    else:
        o = torch.empty_like(q)
    # 在 call_flash_attention_v3 中动态设置 BLOCK_M
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "USE_FP8": use_fp8,
        "num_warps": 8,
        "num_stages": 3
    }

    grid = (triton.cdiv(q.size(0), config['BLOCK_M']),)
    flash_attention_v3[grid](
        q, k, v, o,
        q.size(0), q.size(-1),
        q.stride(1), q.stride(0),
        k.stride(1), k.stride(0),
        v.stride(1), v.stride(0),
        **config
    )
    return o

import torch.nn as nn
import torch.nn.functional as F
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query: [batch_size, num_heads, seq_len, d_k]
        # key:   [batch_size, num_heads, seq_len, d_k]
        # value: [batch_size, num_heads, seq_len, d_v]
        # mask:  [batch_size, 1, 1, seq_len] (optional)

        d_k = query.size(-1)  # 获取特征维度
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))  # 计算注意力分数
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 应用mask（将mask位置设为极大负值）

        attn_weights = F.softmax(scores, dim=-1)  # 计算注意力权重
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, value)  # 加权求和
        return output, attn_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model)  # 键变换
        self.W_v = nn.Linear(d_model, d_model)  # 值变换
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # 输入形状: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        # q, k, v: [batch_size, seq_len, d_model]
        # mask:     [batch_size, seq_len] (可选)
        
        # 线性变换 + 分头
        q = self.split_heads(self.W_q(q))  # [batch_size, num_heads, seq_len, d_k]
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # 扩展mask维度
        
        # 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # 合并头部 + 线性变换
        batch_size, _, seq_len, _ = attn_output.size()
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights
    
# --- 全局变量定义 ---
block_size = 1024
n_embd = 64
dropout = 0.1

# 注意力头模块
class Head(nn.Module):
    def __init__(self, head_size, causal=True):
        super().__init__()
        # 三个线性变换层（分别为键、查询、值）
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal
        # 注册一个下三角矩阵 tril，用于实现因果注意力
        if self.causal:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # 获取输入张量 x 的形状(批量大小、时间步长、特征维度)
        k = self.key(x) # 通过键变换层，得到键张量 k
        q = self.query(x) # 通过查询变换层，得到查询张量 q
        wei = q @ k.transpose(-2,-1) * C**-0.5 # 计算注意力权重矩阵 wei，即查询和键的点积，并乘以缩放因子 C**-0.5
        if self.causal:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 将上三角部分的权重设置为负无穷
        wei = F.softmax(wei, dim=-1) # 对权重矩阵进行 Softmax 归一化
        wei = self.dropout(wei) # 应用 Dropout，防止过拟合
        v = self.value(x) # 通过值变换层，得到值张量 v
        out = wei @ v # 根据权重矩阵 wei 对值张量 v 进行加权求和
        return out
    
# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, causal=True):
        super().__init__()
        # 模块列表，每个头独立计算注意力
        self.heads = nn.ModuleList([Head(head_size, causal=causal) for _ in range(num_heads)])
        # 线性变换层，将多个头的输出拼接后投影到目标维度
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 将拼接后的输出通过投影层，将其维度转换为 (B, T, n_embd)并应用dropout
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

def pytorch_attention(q, k, v, causal, head_size):
    # 将输入转换为三维张量 (B=1, T, D)
    x = q.unsqueeze(0)  # 假设输入是单头且batch_size=1
    mha = Head(head_size=head_size, causal=causal)
    mha = mha.to(q.device)
    output = mha(x)
    return output.squeeze(0)  # 移除批次维度


def pytorch_MultiAttention(q, k, v, causal, num_heads):
    # 将输入转换为三维张量 (B=1, T, D)
    x = q.unsqueeze(0)  # 假设输入是单头且batch_size=1
    mha = MultiHeadAttention(num_heads=num_heads, head_size=n_embd//num_heads, causal=causal)
    mha = mha.to(q.device)
    output = mha(x)
    return output.squeeze(0)  # 移除批次维度



import matplotlib.pyplot as plt
import numpy as np
def plot(times, memory_usage):
    # 收集运行时间和显存消耗
    methods = ['Pytorch Attention', 'Triton FlashAttention-v1', 'Triton FlashAttention-v2', 'Triton FlashAttention-v3']

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.35

    # 绘制运行时间柱状图
    plt.bar(x - width/2, times, width, label='time(ms)', color='#1f77b4')  # 使用更柔和的蓝色
    # 绘制显存消耗柱状图
    plt.bar(x + width/2, memory_usage, width, label='Memory Usage (MB)', color='#ff7f0e')  # 使用更柔和的橙色

    plt.xlabel('Method', fontsize=12, fontfamily='sans-serif')  # 设置字体大小和类型
    plt.ylabel('Value', fontsize=12, fontfamily='sans-serif')
    plt.title('Comparison of Test Times and Memory Usage for Different Methods', fontsize=14, fontfamily='sans-serif')

    plt.xticks(x, methods, fontsize=10, fontfamily='sans-serif')  # 设置x轴标签的字体大小和类型
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)  # 将图例放置在图表下方

    # 在柱子上方显示数值
    for i, (acc, mem) in enumerate(zip(times, memory_usage)):
        plt.text(i - width/2, acc + 0.5, f'{acc:.2f}ms', ha='center', va='bottom', fontsize=10, fontfamily='sans-serif')
        plt.text(i + width/2, mem + 0.5, f'{mem:.2f} MB', ha='center', va='bottom', fontsize=10, fontfamily='sans-serif')

    plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线，设置样式和透明度
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    plt.savefig("acc.png", dpi=300)  # 保存图像，设置分辨率
    plt.show()


def benchmark_fn(fn, args, num_repeats=1000):
    # 预热（建议预热次数与迭代次数成比例）
    for _ in range(100):  # 预热次数增加至 100 次
        fn(*args)
    torch.cuda.synchronize()  # 强制同步
    
    # 使用 CUDA Event 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    
    for _ in range(num_repeats):
        # 清理显存
        torch.cuda.empty_cache()
        
        # 记录时间
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()  # 必须同步！
        
        # 计算单次耗时（单位：毫秒）
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    
    # 返回平均时间（过滤异常值）
    times = sorted(times)
    trimmed_times = times[50:-50]  # 去掉前5%和后5%的异常值
    return sum(trimmed_times) / len(trimmed_times)

from timeit import Timer
"""基准测试函数"""
def benchmark_attention():
    torch.manual_seed(42)
    configs = [
        {'seq_len': 1024, 'd_model': 128, 'num_heads': 8},
        {'seq_len': 16392, 'd_model': 512, 'num_heads': 16},
    ]
    
    for cfg in configs:
        global n_embd, block_size
        n_embd, block_size = cfg['d_model'], cfg['seq_len']
        print(f"\nBenchmarking config: {cfg}")
        
        # 生成测试数据（单头）
        q_single = torch.randn(cfg['seq_len'], cfg['d_model'], device='cuda', dtype=torch.float32)
        k_single, v_single = torch.randn_like(q_single), torch.randn_like(q_single)
        head_size = cfg['d_model'] // cfg['num_heads']
        # 生成多头测试数据
        q_multi = torch.randn(cfg['seq_len'], cfg['num_heads'], head_size, device='cuda', dtype=torch.float16)
        k_multi = torch.randn_like(q_multi)
        v_multi = torch.randn_like(q_multi)
        # 测试用例

        test_cases = [
            ('PyTorch Native', pytorch_attention, (q_single, k_single, v_single, False, head_size)),
            ('FlashAttention-v1', call_flash_attention_v1, (q_single, k_single, v_single)),
            ('FlashAttention-v2', call_flash_attention_v2, (q_multi, k_multi, v_multi, False)),
            ('FlashAttention-v3', call_flash_attention_v3, (q_multi, k_multi, v_multi, False)),
        ]

        # 运行基准测试
        results = {}
        mem_usage = {}
        for name, fn, args in test_cases:
            # 预热和显存重置
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # 测量时间（使用修改后的 benchmark_fn）
            time_ms = benchmark_fn(fn, args, num_repeats=1000)
            results[name] = time_ms
            
            # 测量显存（保持原有逻辑）
            fn(*args)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            mem_usage[name] = peak_mem
        
        # 打印结果
        times = []
        memory_usage = []
        print("\nTime (ms):")
        for name, time in results.items():
            times.append(time)
            print(f"{name}: {time:.3f}ms")
        
        print("\nPeak Memory (MB):")
        for name, mem in mem_usage.items():
            memory_usage.append(mem)
            print(f"{name}: {mem:.2f}MB")

        plot(times, memory_usage)

if __name__ == "__main__":
    benchmark_attention()
