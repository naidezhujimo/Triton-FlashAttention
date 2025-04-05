import torch
import triton
import triton.language as tl
from timeit import Timer

@triton.jit
def attention_forward(
    q_ptr,              # 输入Q矩阵指针
    k_ptr,              # 输入K矩阵指针
    v_ptr,              # 输入V矩阵指针
    o_ptr,              # 输出O矩阵指针
    seq_len,            # 序列长度N
    d_model,            # 特征维度D
    stride_qm,          # Q在序列维度的步骤(通常为d_model)
    stride_km,          # K在序列维度的步幅
    stride_vm,          # V在序列维度的步骤
    BLOCK_M: tl.constexpr, # 每个块处理的Q行数
    BLOCK_N: tl.constexpr, # 每个块处理的K列数
    BLOCK_D: tl.constexpr, # 特征维度分块大小
):
    # 获取当前程序处理的块索引
    pid = tl.program_id(axis=0)
    # 计算当前块处理的Q行范围 [start_m:end_m]
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # K/V的列范围 [start_n:end_n]
    offs_n = tl.arange(0, BLOCK_N)
    # 特征维度分块 [0:BLOCK_D]
    offs_d = tl.arange(0, BLOCK_D)

    # 加载Q分块 (seq_len, d_model)
    # 构造Q块的指针偏移量 [BLOCK_M, BLOCK_D]
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :])
    # 加载数据
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0).to(tl.float32)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # 确保对齐
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载k分块 [BLOCK_N, BLOCK_D]
        k_ptrs = k_ptr + (offs_n[:, None] * stride_km + offs_d[None, :])
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0).to(tl.float32)

        # 加载v分块 [BLOCK_N, BLOCK_D]
        v_ptrs = v_ptr + (offs_n[:, None] * stride_vm + offs_d[None, :])
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0).to(tl.float32)

        # 计算QK^T
        s = tl.dot(q, k.T)
        s *= 1.0 / tl.sqrt(tl.cast(d_model, tl.float32))

        # Softmax
        # 减去最大值保证数据稳定
        s_minus_max = s - tl.max(s, axis=1)[:, None]
        numerator = tl.exp(s_minus_max)
        denominator = tl.sum(numerator, axis=1)[:, None] + 1e-6
        p = numerator / denominator

        # 更新累加器
        acc += tl.dot(p, v)

    # 存储结果
    o_ptrs = o_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :])
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model))

def triton_attention(q, k, v):
    # 参数校验
    assert q.shape[1] == k.shape[1] == v.shape[1], "Feature dim mismatch"
    seq_len, d_model = q.shape
    # 输出矩阵初始化
    o = torch.empty_like(q)
    # 定义分块策略
    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 64
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    # 启动内核
    attention_forward[grid](
        q, k, v, o,
        seq_len, d_model,
        q.stride(0), k.stride(0), v.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return o



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
    q = tl.load(q_ptr + q_offset + tl.arange(0, head_dim)[None, :] * 1,
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
            causal_mask = (start_m + offs_m[:, None]) >= (start_n + offs_n[None, :])
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
        'BLOCK_M': 64,
        'BLOCK_N': 64,
        'num_warps': 4,
        'num_stages': 1,
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

    # 初始化累加器
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # 加载当前Q块（保留FP8量化）
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :],
        mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim),
        other=0.0
    ).to(tl.float32)
    if USE_FP8:
        q = q.to(tl.float8e5, bitcast=True).to(tl.float32)

    # 使用块指针加载K/V块
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_km, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(seq_len, head_dim),
        strides=(stride_vm, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, head_dim),
        order=(1, 0)
    )
    curr_k = tl.load(k_block_ptr, boundary_check=(0, 1)).to(tl.float32)
    curr_v = tl.load(v_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    # 主循环
    for start_n in range(0, seq_len, BLOCK_N):
        # 预加载下一个K/V块
        if start_n + BLOCK_N < seq_len:
            next_k_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
            next_v_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))
            next_k = tl.load(next_k_ptr, boundary_check=(0, 1)).to(tl.float32)
            next_v = tl.load(next_v_ptr, boundary_check=(0, 1)).to(tl.float32)
        else:
            next_k = tl.zeros((BLOCK_N, head_dim), dtype=tl.float32)
            next_v = tl.zeros((BLOCK_N, head_dim), dtype=tl.float32)

        # 若启用FP8，量化当前块
        if USE_FP8:
            curr_k = curr_k.to(tl.float8e5, bitcast=True).to(tl.float32)
            curr_v = curr_v.to(tl.float8e5, bitcast=True).to(tl.float32)

        # 计算QK^T（确保维度匹配）
        s = tl.dot(q, tl.trans(curr_k), allow_tf32=True)
        s = s * (1.0 / tl.sqrt(tl.cast(head_dim, tl.float32)))

        # 在线Softmax
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        alpha = tl.exp(m_i - m_curr)
        beta = tl.exp(s - m_curr[:, None])
        l_curr = alpha * l_i + tl.sum(beta, axis=1)
        p = beta / (l_curr[:, None] + 1e-6)

        # 累加输出
        acc = acc * alpha[:, None] + tl.dot(p, curr_v.to(tl.float32))

        # 切换当前块与预加载块
        curr_k, curr_v = next_k, next_v

    # 存储结果
    tl.store(
        o_ptr + offs_m[:, None] * stride_qm + tl.arange(0, head_dim)[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < seq_len) & (tl.arange(0, head_dim)[None, :] < head_dim)
    )



def call_flash_attention_v3(q, k, v, use_fp8=False):
    assert q.dim() == 3, "Input should be [seq_len, num_heads, head_dim]"
    seq_len, num_heads, head_dim = q.shape
    o = torch.empty_like(q)

    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "USE_FP8": use_fp8,
        "num_warps": 4,  # 减少线程束数以适配旧硬件
        "num_stages": 1   # 减少流水线阶段
    }

    grid = (triton.cdiv(seq_len, config['BLOCK_M']), )
    flash_attention_v3[grid](
        q, k, v, o,
        seq_len, head_dim,
        q.stride(1), q.stride(0),
        k.stride(1), k.stride(0),
        v.stride(1), v.stride(0),
        **config
    )
    return o

import torch.nn.functional as F
"""PyTorch原生Attention实现"""
def pytorch_attention(q, k, v, is_causal=False):
    # 兼容单头和多头输入
    if q.dim() == 2:
        # 单头情况：添加头维度
        q, k, v = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    
    # 使用PyTorch内置优化实现（适用于PyTorch 2.0+）
    o = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal
    )
    
    # 移除多余的头维度（如果原始输入是单头）
    if q.shape[1] == 1:
        o = o.squeeze(1)
    return o

"""基准测试函数"""
def benchmark_attention():
    torch.manual_seed(42)
    
    # 测试配置
    configs = [
        # 更大特征维度测试
        {'seq_len': 8192, 'd_model': 256, 'num_heads': 8, 'head_dim': 128},
        # 极限长序列测试
        {'seq_len': 32768, 'd_model': 256, 'num_heads': 8, 'head_dim': 128},
    ]
    
    for cfg in configs:
        print(f"\nBenchmarking config: {cfg}")
        
        # 生成单头测试数据
        q_single = torch.randn(cfg['seq_len'], cfg['d_model'], device='cuda', dtype=torch.float16)
        k_single = torch.randn_like(q_single)
        v_single = torch.randn_like(q_single)
        
        # 生成多头测试数据
        q_multi = torch.randn(cfg['seq_len'], cfg['num_heads'], cfg['head_dim'], device='cuda', dtype=torch.float16)
        k_multi = torch.randn_like(q_multi)
        v_multi = torch.randn_like(q_multi)
        
        # 测试函数列表
        test_cases = [
            ('PyTorch Native', pytorch_attention, (q_single, k_single, v_single, False)),
            ('Original Triton', triton_attention, (q_single, k_single, v_single)),
            ('FlashAttention-v1', call_flash_attention_v1, (q_single, k_single, v_single)),
            ('FlashAttention-v2', call_flash_attention_v2, (q_multi, k_multi, v_multi, False)),
            ('FlashAttention-v3', call_flash_attention_v3, (q_multi, k_multi, v_multi, False)),
        ]

        # 运行基准测试
        results = {}
        mem_usage = {}
        for name, fn, args in test_cases:
            # 预热
            for _ in range(10):
                fn(*args)
            torch.cuda.synchronize()
            
            # 测量时间
            timer = Timer(
                stmt='fn(*args)',
                globals={'fn': fn, 'args': args},
            )
            total_time_seconds = timer.timeit(100)
            time_ms = (total_time_seconds / 100) * 1000
            results[name] = time_ms
            
            # 测量显存
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            fn(*args)
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # 转换为MB
            mem_usage[name] = peak_mem
        
        # 打印结果
        print("\nTime (ms):")
        for name, time in results.items():
            print(f"{name}: {time:.3f}ms")
        
        print("\nPeak Memory (MB):")
        for name, mem in mem_usage.items():
            print(f"{name}: {mem:.2f}MB")
            
if __name__ == "__main__":
    benchmark_attention()