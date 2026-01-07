# 🎙️ AI Voice Assistant (v2.0)

基于 Python 的全栈语音交互助手，集成 ASR、TTS、LLM、情感识别、声纹识别和语音增强功能。支持 **单机模式** 和 **客户端-服务器** 架构。

## ✨ 核心特性

- **🧠 多模态交互**: 集成 ASR (FunASR), TTS (Coqui/Edge), LLM (Qwen/OpenAI)。
- **🎭 情感识别**: 实时分析语音情感，让回复更有温度。
- **🔐 声纹识别**: 基于 ECAPA-TDNN 的说话人识别与验证。
- **🔊 语音增强**: 包含降噪、AGC (自动增益控制) 和 VAD (语音活动检测)。
- **🛠️ 灵活架构**: 
  - **单机模式**: 适合 PC 直接运行。
  - **C/S 架构**: 适合树莓派作为瘦客户端，PC 作为计算服务器。
- **📦 丰富插件**: 日程管理、新闻播报、天气查询、节日提醒、家庭留言板。

## 📂 项目结构

```
voice/
├── main.py                 # 🚀 单机模式启动入口
├── app/
│   ├── core/               # 核心 AI 模块 (ASR, TTS, LLM, Emotion, Speaker)
│   ├── features/           # 功能插件 (Schedule, News, Weather...)
│   ├── server/             # WebSocket 服务器代码
│   └── client/             # WebSocket 客户端代码
├── data/                   # 数据存储 (声纹库, 配置, 日程数据)
├── tools/                  # 实用工具 (声纹注册, 模型下载)
├── models/                 # 本地 AI 模型权重
└── requirements.txt        # 项目依赖
```

## 🚀 快速开始

### 1. 环境准备

推荐使用 Python 3.10+ 和虚拟环境：

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

> **注意**: 部分音频库（如 PyAudio）可能需要系统级依赖：
> - macOS: `brew install portaudio`
> - Linux: `sudo apt-get install portaudio19-dev`

### 2. 配置

修改 `app/core/config.py` 配置你的 API Key（如果使用云端 LLM）：

```python
LLM_API_KEY = "你的API密钥"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1" # 或其他 OpenAI 兼容接口
```

### 3. 下载模型

项目依赖本地 ASR 模型。请运行下载脚本：

```bash
python tools/download.py
```

### 4. 运行

#### 方式一：单机模式 (推荐测试)

直接在 PC 上运行完整的语音助手：

```bash
python main.py
```

#### 方式二：客户端-服务器模式 (部署)

**服务端 (PC):**
```bash
python app/server/server.py
```

**客户端 (树莓派):**
修改 `app/client/client_config.py` 中的 `SERVER_HOST` 为服务端 IP，然后运行：
```bash
python app/client/client.py
```

## 🎤 声纹注册

要启用个性化功能（如留言板），需要先注册声纹：

```bash
python tools/register_speaker.py
```
按照提示操作，录制 3 段语音即可完成注册。

## 🔌 功能模块

| 模块 | 触发指令示例 | 说明 |
|------|--------------|------|
| **日程管理** | "帮我记明天早上8点开会" | 支持添加、查询、删除日程 |
| **天气查询** | "今天天气怎么样" | 需配置心知天气 API Key |
| **新闻播报** | "有什么新闻" | 获取最新简报 |
| **留言板** | "给张三留言..." | 自动识别说话人并存取留言 |
| **节日提醒** | (自动触发) | 特定节日自动问候 |

## 🛠️ 高级配置

- **语音增强**: 在 `app/core/config.py` 设置 `ENHANCEMENT_MODE` ('basic'/'advanced')
- **ASR/TTS**: 可切换不同模型路径或后端

## 📄 许可证

MIT License
