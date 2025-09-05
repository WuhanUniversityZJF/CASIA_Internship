import subprocess
import os
import torch
import GPUtil

def select_least_used_gpu():
    if not torch.cuda.is_available():
        print("[Error] No CUDA device available.")
        return None

    gpus = GPUtil.getGPUs()
    if not gpus:
        print("[Error] No GPU detected by GPUtil.")
        return None

    # 选择显存使用率最低的 GPU
    sorted_gpus = sorted(gpus, key=lambda x: x.memoryUtil)
    best_gpu = sorted_gpus[0]
    print(f"[INFO] Selecting GPU {best_gpu.id} ({best_gpu.name}) with memory usage: {best_gpu.memoryUsed}/{best_gpu.memoryTotal} MB")
    return best_gpu.id

def main():
    best_gpu_id = select_least_used_gpu()
    if best_gpu_id is None:
        print("[INFO] Falling back to CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)

    print(f"[INFO] CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

    # 启动主脚本
    script_path = "C:\\Users\\Dobot\\Desktop\\Lenna-main\\Lenna-main\\chat_save_mem.py"  # 修改为你的脚本路径
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run script: {e}")

if __name__ == "__main__":
    main()
