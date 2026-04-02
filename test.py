import torch
print(f"版本号: {torch.__version__}")
print(f"显卡是否识别: {torch.cuda.is_available()}")
print(f"显卡型号: {torch.cuda.get_device_name(0)}")
import torch
import time

# 设置矩阵大小 (10000x10000 的 float32 矩阵大约占用 400MB 显存)
size = 1000
dtype = torch.float16

# 1. CPU 测试
print(f"--- 正在初始化 {size}x{size} 矩阵 (CPU) ---")
a_cpu = torch.randn(size, size, dtype=dtype)
b_cpu = torch.randn(size, size, dtype=dtype)

start_time = time.time()
c_cpu = torch.matmul(a_cpu, b_cpu)
cpu_time = time.time() - start_time
print(f"CPU 计算耗时: {cpu_time:.4f} 秒")

print("\n" + "=" * 40 + "\n")

# 2. GPU 测试
if torch.cuda.is_available():
    print(f"--- 正在初始化 {size}x{size} 矩阵 (GPU: {torch.cuda.get_device_name(0)}) ---")

    # 将数据搬运到显存
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()

    # GPU 预热 (Warm-up)：第一次调用 GPU 往往会有初始化开销，不计入正式成绩
    torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    start_time = time.time()

    # 正式计算
    c_gpu = torch.matmul(a_gpu, b_gpu)

    # 关键步骤：等待 GPU 所有计算完成
    torch.cuda.synchronize()

    gpu_time = time.time() - start_time
    print(f"GPU 计算耗时: {gpu_time:.4f} 秒")
    print(f"GPU 相比 CPU 快了: {cpu_time / gpu_time:.2f} 倍")
else:
    print("未检测到可用 GPU")
