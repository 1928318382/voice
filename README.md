
# 语音对话助手（ASR + LLM + 本地 Coqui TTS）

本项目实现了一个本地运行的中文语音对话系统，包含：

- 语音识别（ASR）：使用 FunASR + ModelScope，支持普通话识别  
- 大模型对话（LLM）：通过 HTTP API 调用（可接本地或云端模型）  
- 文本转语音（TTS）：使用本地 **Coqui TTS 中文模型**，完全离线合成  
- 状态指示：通过“状态灯”提示录音、思考、空闲等状态  

> 当前版本已经去掉 EDGE-TTS，**所有 TTS 均使用本地 Coqui 模型**。

---

## 1. 项目结构（简要）

项目根目录大致包含以下文件（只列含义）：

- main.py：主入口，负责录音 → ASR → LLM → TTS 的整体流程  
- config.py：统一配置（路径、模型 ID、LLM 地址等）  
- asr.py：ASR 引擎封装（FunASR）  
- tts.py：TTS 引擎封装（本地 Coqui TTS，多进程）  
- llm.py：LLM 调用逻辑封装（HTTP API）  
- audio_io.py：录音 / 播放相关逻辑  
- led.py：状态灯相关逻辑  
- download.py：可选的模型预下载脚本  
- fix_tts_config_and_test.py：用于修补 Coqui TTS 配置并测试合成  
- models/：存放本地模型的目录（ASR 缓存 + Coqui TTS 模型）

---

## 2. 环境准备

### 2.1 Python 与 Conda 环境（推荐）

建议使用 Python 3.10 和 Conda 虚拟环境，避免与其他项目冲突。

在终端中执行：

conda create -n voice python=3.10
conda activate voice

### 2.2 安装依赖

项目建议使用 requirements.txt 管理依赖。

在虚拟环境 voice 中执行：

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
 
如未提供 requirements.txt，依赖主要包括（仅列出名称，实际版本参考你的当前环境）：

- PyTorch（包含 torch 与 torchaudio，CPU 或 GPU 版本均可）  
- coqui-tts（Coqui TTS）  
- funasr、modelscope、onnxruntime（ASR）  
- numpy（建议 < 2.0）  
- 音频相关包：librosa、soundfile、sounddevice 或 pyaudio  
- LLM 调用相关：requests、openai（如使用兼容 OpenAI 协议的接口）  

如安装 pyaudio 困难，可只使用 sounddevice，并在项目中统一使用它录音。

---

## 3. 模型准备

### 3.1 ASR 模型（FunASR）

ASR 使用 ModelScope 上的中文模型，模型 ID 配置在 config.py 中（例如 Paraformer 大模型）。  
第一次运行时，FunASR 会通过网络自动下载模型并缓存到本地。

你可以选择：

- 在第一次运行 main.py 时自动下载，或者  
- 预先运行 download.py（如果项目提供）进行缓存。

预下载示例命令：

conda activate voice
python download.py

若未提供 download.py，直接运行 python main.py 也会在第一次调用 ASR 时自动下载。

### 3.2 TTS 模型（Coqui 中文模型）

TTS 使用 Coqui 的中文模型，目录结构要求如下（示意）：

- 项目根目录下的 models/ 目录中，包含：

  - models/tts_models--zh-CN--baker--tacotron2-DDC-GST/
    - model_file.pth
    - config.json
    - scale_stats.npy
    - （后续脚本会生成）config_local.json

你需要：

1. 在本地准备好上述三个原始文件：  
   - model_file.pth  
   - config.json  
   - scale_stats.npy  
2. 确保它们放在项目根目录下的 models/tts_models--zh-CN--baker--tacotron2-DDC-GST/ 中。

---

## 4. 修补 Coqui TTS 配置（生成 config_local.json）

许多 Coqui 模型的原始 config.json 内，stats_path（或类似字段）会指向原作者机器上的绝对路径（例如 /home/.../scale_stats.npy），在你本地会导致找不到文件。

本项目提供了 fix_tts_config_and_test.py 用于自动修补：

步骤：

1. 确保 models/tts_models--zh-CN--baker--tacotron2-DDC-GST/ 中已有：
   - model_file.pth
   - config.json
   - scale_stats.npy

2. 在项目根目录运行：

conda activate voice
python fix_tts_config_and_test.py

运行成功后：

- 会在同一目录生成 config_local.json  
- 会在项目根目录生成一个测试音频文件（例如 coqui_test.wav）  

此时，本地 Coqui TTS 模型已可正常使用。

---

## 5. 配置说明（config.py）

项目中的 config.py 用于配置路径与参数，包括：

- 项目根目录与 models/ 路径  
- ASR 模型 ID（FunASR / ModelScope）  
- TTS 模型路径（Coqui，本地文件）  
- LLM 服务的 Base URL、模型名称、API Key 环境变量名等  
- 采样率、超时时间、是否启用某些兜底逻辑等  

若你需要在新环境中复现当前版本，请确保：

1. config.py 中的 TTS 配置指向修补后的配置文件：
   - TTS 模型路径指向 model_file.pth  
   - TTS 配置路径指向 config_local.json（不是原始的 config.json）  

2. LLM 配置与当前使用的后端一致：
   - 若使用本地 LLM，请配置对应的 base_url 和 model 名字  
   - 若使用在线服务，请根据服务商要求设置 API Key（通常从环境变量读取）

---

## 6. 运行项目

### 6.1 预下载（可选）

如有 download.py，可以先执行预下载：

conda activate voice
python download.py

### 6.2 测试 TTS 模块（可选）

若想单独测试 TTS，可以运行（按你当前版本的 tts.py 支持情况）：

conda activate voice
python tts.py

一般会输出一段测试语音，用于确认 Coqui 本地模型与播放逻辑均正常。

### 6.3 启动完整语音对话系统

在项目根目录执行：

conda activate voice
python main.py

典型交互流程：

1. 终端提示你按键开始录音；  
2. 录音结束后，系统显示“正在识别”，然后输出 ASR 结果；  
3. 系统将文本传给 LLM，生成回复；  
4. 回复文本交给 Coqui TTS 合成和播放；  
5. 状态灯在不同阶段切换（录音 / 思考 / 播放 / 空闲）。

---

## 7. 日志与常见现象说明

- 第一次运行 TTS 时，可能会看到 jieba 相关的日志（构建前缀词典），属正常现象。  
- 如日志中出现“某某字符不在词表中，将被丢弃”的 TTS Warning，多为颜文字或 emoji，不影响整体播放。  
- 若 LLM 调用失败（如网络中断），当前版本会返回一条“兜底回复”，并继续由 TTS 播放。

---

## 8. 复现当前版本的最小命令列表

在一台新机器上，从零复现当前可运行版本的大致步骤如下：


我现在将使用uv管理器解决环境依赖问题

pip install uv

uv sync --frozen

source .venv/bin/activate

还有一些系统级环境，如ffmpeg，报错就下即可，有的要从官网下

3. 准备模型文件夹（拷贝或解压得到）：

- 将 models/tts_models--zh-CN--baker--tacotron2-DDC-GST/ 放到项目根目录下  
- 确保其中包含 model_file.pth、config.json、scale_stats.npy

4. 运行配置修补与 TTS 测试脚本：

uv run python fix_tts.py

5. （可选）预下载 ASR 模型：

uv run python download.py

6. 启动主程序：

uv run python main.py

若以上步骤均成功，便可在新环境中完整复现你当前版本的语音对话助手。
