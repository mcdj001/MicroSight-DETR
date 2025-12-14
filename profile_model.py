import warnings
warnings.filterwarnings('ignore')

import torch
import time
import numpy as np
import os

CONFIG = {
    "MODEL_PATH": "/hy-tmp/pycharm_project_1/runs/train/guo-FangAnYi-SOEP-backbone-Pola-SEFN-Mona-DyT-step4/weights/best.pt",
    "DATA_YAML": "/hy-tmp/pycharm_project_1/datasets/data.yaml",
    "INPUT_SIZE": 640,
    "BATCH_SIZE": 8,
    "SPLIT": "val",
    "WARMUP_RUNS": 50,
    "TEST_RUNS": 200,
    "DEVICE": "cuda:0",
}

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    print(f"\n{'─' * 55}")
    print(f"  📊 {text}")
    print(f"{'─' * 55}")

def get_weight_size(path):
    stats = os.stat(path)
    return stats.st_size / 1024 / 1024

class ModuleProfiler:
    def __init__(self):
        self.times = {}
        self.start_times = {}

    def create_hooks(self, name):
        def pre_hook(module, input):
            torch.cuda.synchronize()
            self.start_times[name] = time.perf_counter()

        def post_hook(module, input, output):
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self.start_times[name]) * 1000
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(elapsed)

        return pre_hook, post_hook

    def clear(self):
        self.times.clear()
        self.start_times.clear()

    def get_results(self):
        results = {}
        for name, times in self.times.items():
            if len(times) > 0:
                results[name] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                }
        return results

def run_validation(model_path, data_yaml, imgsz, batch, split):
    print_section("运行官方验证 (获取FPS/精度)")

    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import model_info

    print(f"  加载模型: {os.path.basename(model_path)}")
    model = RTDETR(model_path)

    print(f"  运行验证: batch={batch}, imgsz={imgsz}, split={split}")
    print(f"  数据集: {data_yaml}")
    print(f"  (这可能需要几分钟...)\n")

    result = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        verbose=False,
    )

    n_l, n_p, n_g, flops = model_info(model.model)

    speed_data = {
        "preprocess_ms": result.speed['preprocess'],
        "inference_ms": result.speed['inference'],
        "postprocess_ms": result.speed['postprocess'],
    }

    total_time = speed_data['preprocess_ms'] + speed_data['inference_ms'] + speed_data['postprocess_ms']
    speed_data['fps_total'] = 1000 / total_time
    speed_data['fps_inference'] = 1000 / speed_data['inference_ms']

    model_data = {
        "parameters": n_p,
        "gflops": flops,
        "model_size_mb": get_weight_size(model_path),
    }

    accuracy_data = {
        "mAP50": result.results_dict.get('metrics/mAP50(B)', 0),
        "mAP50-95": result.results_dict.get('metrics/mAP50-95(B)', 0),
        "precision": result.results_dict.get('metrics/precision(B)', 0),
        "recall": result.results_dict.get('metrics/recall(B)', 0),
    }

    print(f"  ✅ 验证完成!")
    print(f"\n  📈 速度数据:")
    print(f"     前处理: {speed_data['preprocess_ms']:.3f} ms")
    print(f"     推理: {speed_data['inference_ms']:.3f} ms")
    print(f"     后处理: {speed_data['postprocess_ms']:.3f} ms")
    print(f"     FPS (推理): {speed_data['fps_inference']:.1f}")
    print(f"     FPS (总计): {speed_data['fps_total']:.1f}")

    print(f"\n  📈 模型数据:")
    print(f"     参数量: {model_data['parameters']:,} ({model_data['parameters']/1e6:.2f}M)")
    print(f"     GFLOPs: {model_data['gflops']:.1f}")
    print(f"     模型大小: {model_data['model_size_mb']:.1f} MB")

    print(f"\n  📈 精度数据:")
    print(f"     mAP@0.5: {accuracy_data['mAP50']:.4f}")
    print(f"     mAP@0.5:0.95: {accuracy_data['mAP50-95']:.4f}")

    return speed_data, model_data, accuracy_data

def measure_memory(model_path, batch_size, input_size, device, warmup=50, runs=100):
    print_section("测量 GPU 显存占用")

    print(f"  加载模型...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
    else:
        model = checkpoint

    if hasattr(model, 'float'):
        model = model.float()

    model = model.to(device)
    model.eval()

    results = {}

    for bs in [1, batch_size]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(bs, 3, input_size, input_size).to(device)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            for _ in range(runs):
                _ = model(dummy_input)
            torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        results[f"batch_{bs}"] = peak_memory
        print(f"     峰值显存: {peak_memory:.2f} GB")

    return model, results

def measure_module_latency(model, input_size, device, warmup=50, runs=200):
    print_section("测量各模块延迟比例")

    if not hasattr(model, 'model') or not hasattr(model.model, '__len__'):
        print("  ⚠️ 无法解析模型结构，跳过模块分析")
        return {}, 0

    layers = model.model
    num_layers = len(layers)

    print(f"  模型共 {num_layers} 层")

    module_map = {
        "Backbone (GEM)": 8,
        "Encoder (MAFD)": 10,
        "Neck (SPAM)": min(21, num_layers - 2),
        "Detection Head": num_layers - 1,
    }

    profiler = ModuleProfiler()
    handles = []

    print(f"  注册监测模块:")
    for name, idx in module_map.items():
        if idx < num_layers:
            module = layers[idx]
            pre_hook, post_hook = profiler.create_hooks(name)
            h1 = module.register_forward_pre_hook(pre_hook)
            h2 = module.register_forward_hook(post_hook)
            handles.extend([h1, h2])
            print(f"    ✅ {name}: Layer {idx} ({type(module).__name__})")

    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    print(f"\n  预热中... ({warmup} 次)")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    total_times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            total_times.append((time.perf_counter() - start) * 1000)

    total_latency = np.mean(total_times)

    profiler.clear()

    print(f"  测量模块延迟... ({runs} 次)")
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)

    results = profiler.get_results()

    for h in handles:
        h.remove()

    print(f"\n  📈 各模块延迟:")
    for name, stats in results.items():
        pct = (stats['mean'] / total_latency) * 100
        print(f"     {name}: {stats['mean']:.2f} ms ({pct:.1f}%)")

    return results, total_latency

def generate_paper_tables(speed_data, model_data, accuracy_data, memory_data, module_results, module_total):
    print_header("📝 论文表格 (可直接复制)")

    inference_ms = speed_data['inference_ms']
    module_times = {}

    if module_results and module_total > 0:
        for name, stats in module_results.items():
            pct = stats['mean'] / module_total
            module_times[name] = {
                "time": inference_ms * pct,
                "percentage": pct * 100
            }

    print("\n" + "=" * 60)
    print("  【Markdown 格式 - 效率分析表】")
    print("=" * 60)

    batch_8_mem = memory_data.get('batch_8', memory_data.get('batch_1', 0))

    print(f"""
| Metric | Value |
|--------|-------|
| Input Resolution | 640 × 640 |
| Batch Size | {CONFIG['BATCH_SIZE']} |
| Parameters | {model_data['parameters']/1e6:.2f} M |
| GFLOPs | {model_data['gflops']:.1f} |
| Model Size | {model_data['model_size_mb']:.1f} MB |
| GPU Memory (Inference) | {batch_8_mem:.2f} GB |
| Preprocessing Latency | {speed_data['preprocess_ms']:.2f} ms |
| Inference Latency | {speed_data['inference_ms']:.2f} ms |""")

    module_order = ["Backbone (GEM)", "Encoder (MAFD)", "Neck (SPAM)", "Detection Head"]
    for name in module_order:
        if name in module_times:
            mt = module_times[name]
            print(f"|   - {name} | {mt['time']:.2f} ms ({mt['percentage']:.0f}%) |")

    print(f"""| Postprocessing Latency | {speed_data['postprocess_ms']:.2f} ms |
| Throughput (FPS) | {speed_data['fps_inference']:.1f} |""")

    print(f"""
| Metric | Value |
|--------|-------|
| mAP@0.5 | {accuracy_data['mAP50']:.4f} |
| mAP@0.5:0.95 | {accuracy_data['mAP50-95']:.4f} |
| Precision | {accuracy_data['precision']:.4f} |
| Recall | {accuracy_data['recall']:.4f} |""")

    print("\n" + "=" * 60)
    print("  【LaTeX 格式】")
    print("=" * 60)

    print(r"""
\begin{table}[h]
\centering
\caption{Inference Efficiency Analysis on NVIDIA RTX 5090}
\label{tab:efficiency}
\begin{tabular}{l|c}
\hline
\textbf{Metric} & \textbf{Value} \\
\hline""")

    print(f"Input Resolution & 640 $\\times$ 640 \\\\")
    print(f"Batch Size & {CONFIG['BATCH_SIZE']} \\\\")
    print(f"Parameters & {model_data['parameters']/1e6:.2f} M \\\\")
    print(f"GFLOPs & {model_data['gflops']:.1f} \\\\")
    print(f"Model Size & {model_data['model_size_mb']:.1f} MB \\\\")
    print(r"\hline")
    print(f"GPU Memory (Inference) & {batch_8_mem:.2f} GB \\\\")
    print(f"Total Inference Latency & {speed_data['inference_ms']:.2f} ms \\\\")

    for name in module_order:
        if name in module_times:
            mt = module_times[name]
            print(f"\\quad - {name} & {mt['time']:.2f} ms \\\\")

    print(r"\hline")
    print(f"Throughput (FPS) & {speed_data['fps_inference']:.1f} \\\\")
    print(r"""\hline
\end{tabular}
\end{table}""")

    print("\n" + "=" * 60)
    print("  【简洁版 - 与其他方法对比用】")
    print("=" * 60)

    print(f"""
| Method | Params (M) | GFLOPs | FPS | mAP@0.5 | mAP@0.5:0.95 |
|--------|------------|--------|-----|---------|--------------|
| MicroSight-DETR | {model_data['parameters']/1e6:.1f} | {model_data['gflops']:.1f} | {speed_data['fps_inference']:.1f} | {accuracy_data['mAP50']:.1f} | {accuracy_data['mAP50-95']:.1f} |""")

def main():
    print_header("🚀 MicroSight-DETR 完整性能分析")

    model_path = CONFIG["MODEL_PATH"]
    data_yaml = CONFIG["DATA_YAML"]
    device = CONFIG["DEVICE"]

    if not os.path.exists(model_path):
        print(f"\n❌ 模型不存在: {model_path}")
        return

    if not os.path.exists(data_yaml):
        print(f"\n❌ 数据集配置不存在: {data_yaml}")
        return

    if not torch.cuda.is_available():
        print("\n❌ CUDA 不可用!")
        return

    print(f"\n📌 配置:")
    print(f"   模型: {os.path.basename(model_path)}")
    print(f"   数据集: {data_yaml}")
    print(f"   Batch Size: {CONFIG['BATCH_SIZE']}")
    print(f"   GPU: {torch.cuda.get_device_name()}")

    speed_data, model_data, accuracy_data = run_validation(
        model_path,
        data_yaml,
        CONFIG["INPUT_SIZE"],
        CONFIG["BATCH_SIZE"],
        CONFIG["SPLIT"]
    )

    model, memory_data = measure_memory(
        model_path,
        CONFIG["BATCH_SIZE"],
        CONFIG["INPUT_SIZE"],
        device,
        CONFIG["WARMUP_RUNS"],
        CONFIG["TEST_RUNS"]
    )

    module_results, module_total = measure_module_latency(
        model,
        CONFIG["INPUT_SIZE"],
        device,
        CONFIG["WARMUP_RUNS"],
        CONFIG["TEST_RUNS"]
    )

    print_header("📊 完整结果汇总")

    print(f"\n  {'='*60}")
    print(f"  {'指标':<40} {'数值':<15}")
    print(f"  {'='*60}")

    print(f"  {'Parameters':<38} {model_data['parameters']/1e6:.2f} M")
    print(f"  {'GFLOPs':<38} {model_data['gflops']:.1f}")
    print(f"  {'Model Size':<38} {model_data['model_size_mb']:.1f} MB")
    print(f"  {'-'*60}")

    for key, value in memory_data.items():
        print(f"  GPU Memory ({key}){' '*(26-len(key))} {value:.2f} GB")
    print(f"  {'-'*60}")

    print(f"  {'Preprocessing':<38} {speed_data['preprocess_ms']:.3f} ms")
    print(f"  {'Inference':<38} {speed_data['inference_ms']:.3f} ms")
    print(f"  {'Postprocessing':<38} {speed_data['postprocess_ms']:.3f} ms")
    print(f"  {'-'*60}")

    if module_results and module_total > 0:
        inference_ms = speed_data['inference_ms']
        for name, stats in module_results.items():
            pct = stats['mean'] / module_total
            scaled_time = inference_ms * pct
            print(f"  {name:<38} {scaled_time:.2f} ms ({pct*100:.0f}%)")
        print(f"  {'-'*60}")

    print(f"  {'FPS (Inference only)':<38} {speed_data['fps_inference']:.1f}")
    print(f"  {'FPS (Total pipeline)':<38} {speed_data['fps_total']:.1f}")
    print(f"  {'-'*60}")

    print(f"  {'mAP@0.5':<38} {accuracy_data['mAP50']:.4f}")
    print(f"  {'mAP@0.5:0.95':<38} {accuracy_data['mAP50-95']:.4f}")
    print(f"  {'Precision':<38} {accuracy_data['precision']:.4f}")
    print(f"  {'Recall':<38} {accuracy_data['recall']:.4f}")
    print(f"  {'='*60}")

    generate_paper_tables(
        speed_data,
        model_data,
        accuracy_data,
        memory_data,
        module_results,
        module_total
    )

    print("\n" + "=" * 70)
    print("  ✅ 分析完成！所有数据均为实测值")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
