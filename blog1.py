import torch
import torch.nn as nn
import torch.nn.functional as F

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


import triton
import triton.language as tl

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
def multi_head_attention_forward(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, d_model, num_heads,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(seq_len, BLOCK_M)
    head_id = pid // num_pid_m  # 头索引
    pid_m = pid % num_pid_m     # 序列块索引

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = head_id

    # 加载Q分块 [BLOCK_M, BLOCK_D]
    q_ptrs = q_ptr + (offs_m[:, None] * stride_qm + 
                     offs_h * stride_qh + 
                     offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, 
               mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D), 
               other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # 加载K分块 [BLOCK_N, BLOCK_D]
        k_ptrs = k_ptr + (offs_n[:, None] * stride_km + 
                         offs_h * stride_kh + 
                         offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs,
                   mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
                   other=0.0)

        # 计算注意力分数（使用head_dim缩放）
        s = tl.dot(q, k.T) * (1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32)))
        # 因果掩码
        s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float('-inf'))

        # Softmax
        s_max = tl.max(s, axis=1)
        s = s - s_max[:, None]
        numerator = tl.exp(s)
        denominator = tl.sum(numerator, axis=1)[:, None] + 1e-6
        p = numerator / denominator

        # 加载V分块并累加
        v_ptrs = v_ptr + (offs_n[:, None] * stride_vm + 
                         offs_h * stride_vh + 
                         offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs,
                   mask=(offs_n[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D),
                   other=0.0)
        acc += tl.dot(p, v)

    # 存储结果到当前头对应的位置
    o_ptrs = o_ptr + (offs_m[:, None] * stride_qm + 
                     offs_h * stride_qh + 
                     offs_d[None, :] * stride_qd)
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < BLOCK_D))

def triton_multi_head_attention(q, k, v, num_heads):
    assert q.shape == k.shape == v.shape
    seq_len, d_model = q.shape
    head_dim = d_model // num_heads
    
    # 重塑为多头格式 [seq_len, num_heads, head_dim]
    q_multi = q.view(seq_len, num_heads, head_dim).contiguous()
    k_multi = k.view(seq_len, num_heads, head_dim).contiguous()
    v_multi = v.view(seq_len, num_heads, head_dim).contiguous()
    o_multi = torch.empty_like(q_multi)


    # 配置分块参数
    BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, head_dim
    grid = (num_heads * triton.cdiv(seq_len, BLOCK_M),)
    
    multi_head_attention_forward[grid](
        q_multi, k_multi, v_multi, o_multi,
        seq_len, d_model, num_heads,
        q_multi.stride(0), q_multi.stride(1), q_multi.stride(2),
        k_multi.stride(0), k_multi.stride(1), k_multi.stride(2),
        v_multi.stride(0), v_multi.stride(1), v_multi.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return o_multi.view(seq_len, d_model)


import matplotlib.pyplot as plt
import numpy as np
def plot(times, memory_usage):
    # 收集运行时间和显存消耗
    methods = ['Pytorch Attention', 'Pytorch MultiAttention', 'Triton Attention', 'Triton MultiAttention']

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


from timeit import Timer
"""基准测试函数"""
def benchmark_attention():
    torch.manual_seed(42)
    configs = [
        {'seq_len': 1024, 'd_model': 128, 'num_heads': 8},
        {'seq_len': 8096, 'd_model': 128, 'num_heads': 8},
    ]
    
    for cfg in configs:
        global n_embd, block_size
        n_embd, block_size = cfg['d_model'], cfg['seq_len']
        print(f"\nBenchmarking config: {cfg}")
        
        # 生成测试数据（单头）
        q_single = torch.randn(cfg['seq_len'], cfg['d_model'], device='cuda', dtype=torch.float32)
        k_single, v_single = torch.randn_like(q_single), torch.randn_like(q_single)
        
        # 测试用例
        head_size = cfg['d_model'] // cfg['num_heads']
        test_cases = [
            ('PyTorch Attention', pytorch_attention, (q_single, k_single, v_single, False, head_size)),
            ('PyTorch MultiAttention', pytorch_MultiAttention, (q_single, k_single, v_single, False, cfg['num_heads'])),
            ('Triton Attention', triton_attention, (q_single, k_single, v_single)),
            ('Triton MultiAttention', triton_multi_head_attention, (q_single, k_single, v_single, cfg['num_heads'])),
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