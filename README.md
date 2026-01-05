
# 语音对话助手（ASR + LLM + 本地 Coqui TTS）

本项目实现了一个本地运行的中文语音对话系统，包含：

- 语音识别（ASR）：使用 FunASR + ModelScope，支持普通话识别
- 语音增强：实时降噪、自动增益控制、语音活动检测，提升识别准确率
- 大模型对话（LLM）：通过 HTTP API 调用（可接本地或云端模型）
- 文本转语音（TTS）：使用本地 **Coqui TTS 中文模型**，完全离线合成
- 状态指示：通过"状态灯"提示录音、思考、空闲等状态  

> 当前版本已经去掉 EDGE-TTS，**所有 TTS 均使用本地 Coqui 模型**。

---

## 1. 项目结构（简要）

项目根目录大致包含以下文件（只列含义）：

- main.py：主入口（薄包装），负责启动语音助手主流程
- register_speaker.py：声纹注册工具入口（薄包装）
- voice_app/：核心代码（ASR / LLM / TTS / 声纹 / 情感 / 语音增强 / 硬件抽象）
- scripts/：辅助脚本（下载 ASR、修补 TTS 配置、检查麦克风、TTS 测试等）
- tests/：测试脚本
- emotional/：情感识别模型与特征提取代码
- models/：存放本地模型的目录（ASR 缓存 + Coqui TTS 模型 + 声纹模型）

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
- **语音增强包**：noisereduce（降噪）、webrtcvad（语音检测）
- **声纹识别包**：speechbrain（ECAPA-TDNN）、scikit-learn（相似度计算）
- LLM 调用相关：requests、openai（如使用兼容 OpenAI 协议的接口）  

如安装 pyaudio 困难，可只使用 sounddevice，并在项目中统一使用它录音。

---

## 3. 模型准备

### 3.1 ASR 模型（FunASR）

ASR 使用 ModelScope 上的中文模型，模型 ID 配置在 `voice_app/config.py` 中（例如 Paraformer 大模型）。  
第一次运行时，FunASR 会通过网络自动下载模型并缓存到本地。

你可以选择：

- 在第一次运行 main.py 时自动下载，或者  
- 预先运行 scripts/download_asr.py 进行缓存。

预下载示例命令：

conda activate voice
python scripts/download_asr.py

若未执行预下载脚本，直接运行 `python main.py` 也会在第一次调用 ASR 时自动下载（需要网络）。

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

本项目提供了 scripts/fix_tts.py 用于自动修补：

步骤：

1. 确保 models/tts_models--zh-CN--baker--tacotron2-DDC-GST/ 中已有：
   - model_file.pth
   - config.json
   - scale_stats.npy

2. 在项目根目录运行：

conda activate voice
python scripts/fix_tts.py

运行成功后：

- 会在同一目录生成 config_local.json  
- 会在项目根目录生成一个测试音频文件（例如 coqui_test.wav）  

此时，本地 Coqui TTS 模型已可正常使用。

---

## 5. 配置说明（voice_app/config.py）

项目中的 `voice_app/config.py` 用于配置路径与参数，包括：

- 项目根目录与 models/ 路径
- ASR 模型 ID（FunASR / ModelScope）
- TTS 模型路径（Coqui，本地文件）
- **语音增强配置**：增强模式、降噪强度、AGC参数、VAD参数等
- LLM 服务的 Base URL、模型名称、API Key 环境变量名等
- 采样率、超时时间、是否启用某些兜底逻辑等

### 5.1 语音增强配置

语音增强支持4种模式：

- `none`：禁用增强（直通模式）
- `basic`：基础增强（降噪 + AGC）
- `advanced`：高级增强（降噪 + AGC + VAD）
- `full`：完整增强（所有功能，包括预留的深度学习模型）

```python
# 在 voice_app/config.py 中配置
ENHANCEMENT_MODE = "advanced"  # 推荐模式
NOISE_REDUCTION_STRENGTH = 0.8  # 降噪强度 (0-1)
AGC_TARGET_LEVEL = 0.8          # AGC目标电平
VAD_AGGRESSIVENESS = 2          # VAD灵敏度 (0-3)
```  

若你需要在新环境中复现当前版本，请确保：

1. `voice_app/config.py` 中的 TTS 配置指向修补后的配置文件：
   - TTS 模型路径指向 model_file.pth  
   - TTS 配置路径指向 config_local.json（不是原始的 config.json）  

2. LLM 配置与当前使用的后端一致：
   - 若使用本地 LLM，请配置对应的 base_url 和 model 名字  
   - 若使用在线服务，请根据服务商要求设置 API Key（通常从环境变量读取）

---

## 6. 运行项目

### 6.1 预下载（可选）

如有需要，可以先执行预下载：

conda activate voice
python scripts/download_asr.py

### 6.2 测试语音增强模块（可选）

若想单独测试语音增强功能，可以运行：

conda activate voice
python -m voice_app.enhancement demo

这会演示语音增强的效果，包括降噪和音量调节。

### 6.3 测试声纹识别模块（可选）

#### 运行注册工具：
```bash
conda activate voice
python register_speaker.py
```

#### 测试注册功能：
```bash
conda activate voice
python tests/test_registration.py
```

### 6.5 声纹注册指南

#### 启动注册工具：
```bash
conda activate voice
python register_speaker.py
```

#### 在语音助手中注册：
运行主程序后，在键盘输入中使用以下命令：
- `register` - 启动声纹注册工具
- `users` - 查看已注册用户

#### 注册流程：
1. 运行注册工具或在主程序中输入 `register`
2. 选择"1. 交互式注册用户"
3. 输入用户名（字母、数字、下划线）
4. 按照提示录制3个语音样本
5. 每个样本说不同的句子，保持自然语速
6. 注册完成后，系统会显示注册成功的用户信息

#### 注册数据存储：
- 用户声纹特征保存在：`models/speaker/database.pkl`
- 注册信息在程序重启后仍然保留

#### 最佳实践：
- ✅ 选择安静环境录音
- ✅ 使用不同的句子进行注册
- ✅ 每个样本录制3-5秒
- ✅ 保持自然语速和音量
- ✅ 建议注册3-5个样本以提高准确率

### 6.4 测试 TTS 模块（可选）

若想单独测试 TTS，可以运行：

conda activate voice
python scripts/test_tts_local.py

一般会输出一段测试语音，用于确认 Coqui 本地模型与播放逻辑均正常。

### 6.3 启动完整语音对话系统

在项目根目录执行：

conda activate voice
python main.py

典型交互流程：

1. 终端提示你按键开始录音；
2. 录音结束后，系统显示"正在识别"，然后输出 ASR 结果；
3. 系统自动进行声纹识别，显示说话人身份；
4. 系统将文本+情感+声纹信息传给 LLM，生成个性化回复；
5. 回复文本交给 Coqui TTS 合成和播放；
6. 状态灯在不同阶段切换（录音 / 思考 / 播放 / 空闲）。

#### 声纹注册流程：

1. 运行程序后，输入 `register` 启动注册工具
2. 按照提示输入用户名
3. 录制3个语音样本（每个3秒）
4. 注册完成后，输入 `users` 查看注册状态
5. 现在系统就能识别您的声音了！

---

## 7. 日志与常见现象说明

- 第一次运行 TTS 时，可能会看到 jieba 相关的日志（构建前缀词典），属正常现象。  
- 如日志中出现“某某字符不在词表中，将被丢弃”的 TTS Warning，多为颜文字或 emoji，不影响整体播放。  
- 若 LLM 调用失败（如网络中断），当前版本会返回一条“兜底回复”，并继续由 TTS 播放。

---

## 8. 复现当前版本的最小命令列表

在一台新机器上，从零复现当前可运行版本的大致步骤如下：

1. 创建并激活 Conda 环境：

conda create -n voice python=3.10
conda activate voice

2. 安装依赖（假设已有 requirements.txt）：
sudo apt update
sudo apt install build-essential -y
sudo apt install portaudio19-dev libsndfile1 -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install noisereduce webrtcvad librosa --upgrade      # 语音增强依赖
pip install speechbrain scikit-learn huggingface-hub    # 声纹识别依赖
提示：coqui-tts 仅支持 Python 3.9-3.12，如遇 "No matching distribution"，请检查当前 Python 版本。
pip install pandas
sudo apt install ffmpeg
3. 准备模型文件夹（拷贝或解压得到）：

- 将 models/tts_models--zh-CN--baker--tacotron2-DDC-GST/ 放到项目根目录下  
- 确保其中包含 model_file.pth、config.json、scale_stats.npy

4. 运行配置修补与 TTS 测试脚本：

python scripts/fix_tts.py

5. （可选）预下载 ASR 模型：

python scripts/download_asr.py

6. 启动主程序：

python main.py

若以上步骤均成功，便可在新环境中完整复现你当前版本的语音对话助手。
