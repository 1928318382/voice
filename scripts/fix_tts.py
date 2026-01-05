import json
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from TTS.api import TTS

from voice_app.config import BASE_DIR, TTS_MODEL_DIR


def main():
    model_path = os.path.join(TTS_MODEL_DIR, "model_file.pth")
    config_path = os.path.join(TTS_MODEL_DIR, "config.json")
    stats_path = os.path.join(TTS_MODEL_DIR, "scale_stats.npy")

    print("[DEBUG] model_path:", model_path, os.path.exists(model_path))
    print("[DEBUG] config_path:", config_path, os.path.exists(config_path))
    print("[DEBUG] stats_path:", stats_path, os.path.exists(stats_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_file.pth 不存在: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json 不存在: {config_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"scale_stats.npy 不存在: {stats_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    audio_cfg = cfg.get("audio", {})
    audio_cfg["stats_path"] = stats_path
    cfg["audio"] = audio_cfg

    patched_cfg_path = os.path.join(TTS_MODEL_DIR, "config_local.json")
    with open(patched_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("[INFO] 已生成修补后的配置:", patched_cfg_path)

    tts = TTS(
        model_path=model_path,
        config_path=patched_cfg_path,
        progress_bar=False,
        gpu=False,
    )

    out_wav = os.path.join(BASE_DIR, "coqui_test.wav")
    text = "你好，这是使用修补后配置的 Coqui TTS 测试。"

    print("[TTS] 开始合成:", text)
    tts.tts_to_file(text=text, file_path=out_wav)
    print("[TTS] 合成完成，文件已保存到:", out_wav)


if __name__ == "__main__":
    main()

