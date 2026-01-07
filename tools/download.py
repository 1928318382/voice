import os
import shutil
# 引入 modelscope 原生下载函数，它可以精确控制下载目录
# 引入 modelscope 原生下载函数，它可以精确控制下载目录
from modelscope import snapshot_download

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import ASR_MODEL_ID, ASR_MODEL_PATH, ASR_FLAG_FILE


def download_asr():
    print("=" * 50)
    print(f"[下载器] 目标本地路径: {ASR_MODEL_PATH}")
    print("=" * 50)

    # 1. 检查并清理旧目录（确保是纯净的重新下载）
    if os.path.exists(ASR_MODEL_PATH):
        print("   [提示] 检测到目标文件夹已存在，正在清理以确保完整下载...")
        try:
            shutil.rmtree(ASR_MODEL_PATH)
        except Exception as e:
            print(f"   [警告] 清理目录失败 (可能文件被占用): {e}")

    # 2. 执行下载
    print(f"[执行] 正在从 ModelScope 拉取模型: {ASR_MODEL_ID}")
    print("   这可能需要几分钟，取决于你的网速...")

    try:
        # 【核心修改】使用 snapshot_download 并指定 local_dir
        # 这会将云端仓库的所有文件直接下载到 ASR_MODEL_PATH 文件夹内
        snapshot_download(
            model_id=ASR_MODEL_ID,
            local_dir=ASR_MODEL_PATH,
            revision="master"  # 使用默认分支
        )

        print("\n   ✅ 下载成功！所有模型文件已保存在工程目录中。")

        # 3. 写入成功标记
        with open(ASR_FLAG_FILE, "w", encoding="utf-8") as f:
            f.write("ok")
        print(f"   ✅ 已生成标记文件: {ASR_FLAG_FILE}")

        return True

    except Exception as e:
        print(f"\n   ❌ 下载出错: {e}")
        print("   请检查网络连接（如需代理请配置系统代理）。")
        return False


def main():
    if download_asr():
        print("\n🎉 模型已彻底本地化！")
        print(f"   你可以在此处查看文件: {ASR_MODEL_PATH}")
        print("   现在运行 python main.py 将直接使用该文件夹下的模型。")
    else:
        print("\n⚠️ 下载流程未完成。")


if __name__ == "__main__":
    main()