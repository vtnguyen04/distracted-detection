import argparse
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from src.infrastructure.inference_backend import create_backend


def run_benchmark(
    backend_name: str, device: str, model_path: str, use_shm: bool, warmup: int = 15, iterations: int = 50
):
    if backend_name == "openvino" and device.startswith("cuda"):
        print(f"\n⏭️ BỎ QUA: [{backend_name.upper()}] (OpenVINO không hỗ trợ NVIDIA CUDA)")
        return

    mode_str = "SHARED MEMORY (IPC)" if use_shm else "NATIVE RAM"
    print(f"\n🚀 Khởi tạo Benchmark: [{backend_name.upper()}] | Thiết bị: [{device.upper()}] | Bộ nhớ: [{mode_str}]")
    try:
        backend = create_backend(backend_name, model_path, device=device, imgsz=(320, 320))
    except Exception as e:
        print(f"❌ Không thể tải Backend {backend_name}: {e}")
        return

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    shm, shm_array = None, None

    if use_shm:
        # Giả lập chính xác Overhead của kiến trúc Đa Tiến Trình (IPC)
        size = int(np.prod(dummy_frame.shape)) * dummy_frame.dtype.itemsize
        shm = SharedMemory(create=True, size=size)
        shm_array = np.ndarray(dummy_frame.shape, dtype=dummy_frame.dtype, buffer=shm.buf)
        np.copyto(shm_array, dummy_frame)

    print(f"🔥 Bắt đầu Warm-up ({warmup} frames)...")
    for _ in range(warmup):
        frame_to_infer = np.array(shm_array) if use_shm else dummy_frame
        _ = backend.predict(frame_to_infer, confidence=0.4)

    print(f"⚡ Đang Benchmark tốc độ cực hạn ({iterations} frames)...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        # Đo lường BAAO GỒM cả độ trễ đọc từ vùng nhớ dùng chung IPC
        frame_to_infer = np.array(shm_array) if use_shm else dummy_frame
        _ = backend.predict(frame_to_infer, confidence=0.4)
    end_time = time.perf_counter()

    if use_shm:
        shm.close()
        shm.unlink()

    total_time = end_time - start_time
    fps = iterations / total_time
    ms_per_frame = (total_time / iterations) * 1000

    print("=" * 60)
    print(f"🎯 KẾT QUẢ: {backend_name.upper()} ({mode_str})")
    print(f"   ⏱️ Thời gian trung bình: {ms_per_frame:.2f} ms / frame")
    print(f"   🚀 Tốc độ tối đa lý thuyết: {fps:.2f} FPS")
    print("=" * 60)

    backend.release()


def _run_isolated(b: str, dev: str, mod: str, shm: bool) -> None:
    run_benchmark(b, dev, mod, use_shm=shm)


if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/cpu_model/v8n_facedetect_model.onnx")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    backends = ["onnx", "openvino", "ultralytics"]
    if args.device != "cpu":
        backends.append("tensorrt")

    for b in backends:
        p1 = mp.Process(target=_run_isolated, args=(b, args.device, args.model, False))
        p1.start()
        p1.join()

        p2 = mp.Process(target=_run_isolated, args=(b, args.device, args.model, True))
        p2.start()
        p2.join()
