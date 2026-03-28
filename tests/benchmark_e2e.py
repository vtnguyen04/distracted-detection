import argparse
import time

from src.config.settings import AppSettings
from src.engine.process_manager import ProcessManager


def measure_e2e_backend(backend_name: str, device: str, use_shm: bool, duration: int = 15):
    # OpenVINO natively cannot interpret NVIDIA's CUDA cores
    if backend_name == "openvino" and device.startswith("cuda"):
        print(f"\n⏭️ BỎ QUA: [{backend_name.upper()}] (OpenVINO không hỗ trợ NVIDIA CUDA)")
        return

    mode_str = "SHARED MEMORY (MULTI-PROCESS IPC)" if use_shm else "NATIVE RAM (SINGLE-PROCESS THREADED)"
    print(f"\n{'=' * 60}")
    print(f"🚀 BẮT ĐẦU BENCHMARK CAMERA THỰC TẾ: [{backend_name.upper()}] trên [{device.upper()}] | {mode_str}")
    print(f"{'=' * 60}")

    settings = AppSettings()
    # Explicitly toggle Multiprocessing IPC
    settings.pipeline.use_multiprocessing = use_shm
    settings.inference.backend = backend_name
    settings.inference.device = device

    manager = ProcessManager(settings)
    manager.start()

    print("⏳ Đang khởi động Camera và Warm-up Hệ Thống (Chờ 5 giây)...")
    time.sleep(5)

    fps_records = []
    print(f"📡 Đang thu thập chỉ số FPS Pipeline trong {duration} giây...")
    try:
        shm_state = manager._shm.state
        for i in range(duration):
            time.sleep(1.0)
            fps = shm_state.get("fps")
            if fps and fps > 0:
                fps_records.append(fps)
                print(f"   [+] Giây thứ {i + 1:02d}: {fps} FPS")
            else:
                print(f"   [!] Giây thứ {i + 1:02d}: Camera hoặc Model đang kẹt...")
    except KeyboardInterrupt:
        print("\n⚠️ Người dùng ngắt ngang Benchmark.")
    finally:
        print("🛑 Đang đóng luồng Camera và dọn dẹp RAM...")
        manager.stop()

    if fps_records:
        avg_fps = sum(fps_records) / len(fps_records)
        print("\n" + "★" * 50)
        print(f"🎯 KẾT QUẢ ĐĨCH THỰC (PIPELINE E2E): {backend_name.upper()}")
        print(f"   👉 Tốc độ FPS Trung Bình (Bao gồm Camera IO): {avg_fps:.2f} FPS")
        print("★" * 50 + "\n")
    else:
        print(f"\n❌ LỖI: Backend {backend_name.upper()} bị sập, không đo được khung hình nào!")


def _run_e2e_isolated(b: str, dev: str, shm: bool, dur: int):
    measure_e2e_backend(b, dev, shm, dur)


if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    backends = ["onnx", "openvino", "ultralytics"]
    if args.device != "cpu":
        backends.append("tensorrt")

    for b in backends:
        p1 = mp.Process(target=_run_e2e_isolated, args=(b, args.device, False, 15))
        p1.start()
        p1.join()

        p2 = mp.Process(target=_run_e2e_isolated, args=(b, args.device, True, 15))
        p2.start()
        p2.join()
