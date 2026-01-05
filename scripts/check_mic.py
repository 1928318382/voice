import pyaudio


def main():
    p = pyaudio.PyAudio()

    print("=== 当前电脑音频输入设备列表 ===")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")

    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels")) > 0:
            name = p.get_device_info_by_host_api_device_index(0, i).get("name")
            print(f"设备索引 ID: {i} - 名称: {name}")

    print("\n请在 voice_app/config.py 中将 MIC_DEVICE_INDEX 修改为对应的 ID 数字")


if __name__ == "__main__":
    main()

