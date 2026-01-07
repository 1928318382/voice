# fix_tts_config_and_test.py
import os
import json

# 关键：在导入 TTS 之前关闭 weights_only 限制（适配 torch>=2.6）
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from TTS.api import TTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "tts_models--zh-CN--baker--tacotron2-DDC-GST",
)

model_path = os.path.join(MODEL_DIR, "model_file.pth")
config_path = os.path.join(MODEL_DIR, "config.json")
stats_path = os.path.join(MODEL_DIR, "scale_stats.npy")

print("[DEBUG] model_path:", model_path, os.path.exists(model_path))
print("[DEBUG] config_path:", config_path, os.path.exists(config_path))
print("[DEBUG] stats_path:", stats_path, os.path.exists(stats_path))

if not os.path.exists(model_path):
    raise FileNotFoundError(f"model_file.pth 不存在: {model_path}")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"config.json 不存在: {config_path}")
if not os.path.exists(stats_path):
    raise FileNotFoundError(f"scale_stats.npy 不存在: {stats_path}")

# 1) 读取原始 config.json，修改 audio.stats_path -> 指向本机的 scale_stats.npy
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

audio_cfg = cfg.get("audio", {})
audio_cfg["stats_path"] = stats_path
cfg["audio"] = audio_cfg

patched_cfg_path = os.path.join(MODEL_DIR, "config_local.json")
with open(patched_cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("[INFO] 已生成修补后的配置:", patched_cfg_path)

# 2) 用修补后的 config_local.json 跑一遍 TTS 测试
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
