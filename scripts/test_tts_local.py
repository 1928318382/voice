import os

from TTS.api import TTS

from voice_app.config import TTS_MODEL_PATH, TTS_CONFIG_PATH


def main():
    print("MODEL:", TTS_MODEL_PATH, "exists:", os.path.exists(TTS_MODEL_PATH))
    print("CONFIG:", TTS_CONFIG_PATH, "exists:", os.path.exists(TTS_CONFIG_PATH))

    tts = TTS(
        model_path=TTS_MODEL_PATH,
        config_path=TTS_CONFIG_PATH,
        progress_bar=False,
        gpu=False,
    )

    out = "test_local_tts.wav"
    tts.tts_to_file(text="你好，我是本地 TTS 测试。", file_path=out)
    print("done ->", out)


if __name__ == "__main__":
    main()

