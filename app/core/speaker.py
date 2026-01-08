import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torchaudio

# 修复 torchaudio 2.1+ 兼容性问题
# 新版本移除了 list_audio_backends，需要添加兼容层
if not hasattr(torchaudio, 'list_audio_backends'):
    # 添加缺失的函数
    def _list_audio_backends():
        """兼容层：返回可用的音频后端"""
        try:
            # 新版本使用不同的 API
            return list(torchaudio.utils.sox_utils.list_effects()) if hasattr(torchaudio.utils, 'sox_utils') else ['soundfile']
        except:
            return ['soundfile']
    
    torchaudio.list_audio_backends = _list_audio_backends

from sklearn.metrics.pairwise import cosine_similarity
from app.core.config import (
    SPEAKER_MODEL_PATH, SPEAKER_EMBEDDING_DIM, SPEAKER_SAMPLE_RATE,
    SPEAKER_SIMILARITY_THRESHOLD, SPEAKER_DATABASE_PATH,
    SPEAKER_CHUNK_DURATION, SPEAKER_OVERLAP_DURATION, SPEAKER_MIN_AUDIO_LENGTH,
    SPEAKER_RECOGNITION_ENABLED, SPEAKER_MODEL_SOURCE
)


class ECAPATDNNRecognizer:
    """
    基于ECAPA-TDNN的声纹识别器
    支持用户注册、识别和验证
    """

    def __init__(self):
        self.model = None
        self.database: Dict[str, List[np.ndarray]] = {}
        self.device = torch.device("cpu")  # 在树莓派上强制CPU

        if SPEAKER_RECOGNITION_ENABLED:
            print("[Speaker] 正在加载ECAPA-TDNN声纹识别模块...")
            self._load_model()
            self._load_database()
            print("[Speaker] 声纹识别模块加载完毕")
        else:
            print("[Speaker] 声纹识别模块已禁用")

    def _load_model(self):
        """加载ECAPA-TDNN模型"""
        try:
            # 检查是否已安装speechbrain
            try:
                from speechbrain.inference import SpeakerRecognition
                has_speechbrain = True
            except ImportError:
                try:
                     # try old import path for backwards compatibility
                    from speechbrain.pretrained import SpeakerRecognition
                    has_speechbrain = True
                except ImportError:
                    has_speechbrain = False
                    print("[Speaker] SpeechBrain未安装，使用简化实现")

            if has_speechbrain and SPEAKER_MODEL_SOURCE == "speechbrain":
                print("[Speaker] 正在从SpeechBrain加载ECAPA-TDNN模型...")
                # Suppress warnings if needed, but keeping simple for now
                self.model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=SPEAKER_MODEL_PATH,
                    run_opts={"device": self.device}
                )
                print("[Speaker] ECAPA-TDNN模型加载成功")
            else:
                print("[Speaker] 使用简化声纹识别实现 (无预训练模型)")
                self.model = None

        except Exception as e:
            print(f"[Speaker] 模型加载失败: {e}")
            print("[Speaker] 降级为模拟模式")
            self.model = None

    def _load_database(self):
        """加载用户声纹数据库"""
        try:
            if os.path.exists(SPEAKER_DATABASE_PATH):
                with open(SPEAKER_DATABASE_PATH, 'rb') as f:
                    self.database = pickle.load(f)
                print(f"[Speaker] 加载用户数据库: {len(self.database)}个用户")
            else:
                print("[Speaker] 用户数据库不存在，将创建新的数据库")
                self.database = {}
        except Exception as e:
            print(f"[Speaker] 数据库加载失败: {e}")
            self.database = {}

    def _save_database(self):
        """保存用户声纹数据库"""
        try:
            os.makedirs(os.path.dirname(SPEAKER_DATABASE_PATH), exist_ok=True)
            with open(SPEAKER_DATABASE_PATH, 'wb') as f:
                pickle.dump(self.database, f)
            print(f"[Speaker] 数据库已保存: {len(self.database)}个用户")
        except Exception as e:
            print(f"[Speaker] 数据库保存失败: {e}")

    def _preprocess_audio(self, audio_data: bytes) -> torch.Tensor:
        """音频预处理"""
        # 转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 检查音频长度
        min_samples = int(SPEAKER_SAMPLE_RATE * SPEAKER_MIN_AUDIO_LENGTH)
        if len(audio_np) < min_samples:
            # 如果音频太短，填充静音
            padding = np.zeros(min_samples - len(audio_np))
            audio_np = np.concatenate([audio_np, padding])
            print(f"[Speaker] 音频太短，已填充到{SPEAKER_MIN_AUDIO_LENGTH}秒")

        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

        return audio_tensor

    def _extract_embedding(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """提取声纹特征"""
        if self.model is not None:
            # 使用真实的ECAPA-TDNN模型
            try:
                with torch.no_grad():
                    embedding = self.model.encode_batch(audio_tensor)
                    embedding = embedding.squeeze().cpu().numpy()

                # 归一化
                embedding = embedding / np.linalg.norm(embedding)
                return embedding

            except Exception as e:
                print(f"[Speaker] 模型推理失败: {e}")
                return self._extract_embedding_simple(audio_tensor)
        else:
            # 使用简化的特征提取
            return self._extract_embedding_simple(audio_tensor)

    def _extract_embedding_simple(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """简化的声纹特征提取（用于无模型情况）"""
        try:
            audio_np = audio_tensor.squeeze().numpy()

            # 计算MFCC特征作为简化版本
            import librosa
            mfcc = librosa.feature.mfcc(
                y=audio_np,
                sr=SPEAKER_SAMPLE_RATE,
                n_mfcc=40,
                n_fft=1024,
                hop_length=512
            )

            # 取平均值作为embedding
            embedding = np.mean(mfcc, axis=1)

            # 归一化到固定维度
            if len(embedding) < SPEAKER_EMBEDDING_DIM:
                # 填充
                padding = np.zeros(SPEAKER_EMBEDDING_DIM - len(embedding))
                embedding = np.concatenate([embedding, padding])
            elif len(embedding) > SPEAKER_EMBEDDING_DIM:
                # 截断
                embedding = embedding[:SPEAKER_EMBEDDING_DIM]

            # L2归一化
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"[Speaker] 简化特征提取失败: {e}")
            # 返回随机向量作为fallback
            return np.random.randn(SPEAKER_EMBEDDING_DIM)

    def enroll_user(self, user_id: str, audio_data: bytes) -> bool:
        """注册用户声纹"""
        try:
            if len(self.database) >= 20:  # 限制用户数量
                print("[Speaker] 用户数量已达上限")
                return False

            audio_tensor = self._preprocess_audio(audio_data)
            embedding = self._extract_embedding(audio_tensor)

            if user_id not in self.database:
                self.database[user_id] = []

            self.database[user_id].append(embedding)
            self._save_database()

            print(f"[Speaker] 用户{user_id}注册成功，当前样本数: {len(self.database[user_id])}")
            return True

        except Exception as e:
            print(f"[Speaker] 用户注册失败: {e}")
            return False

    def identify(self, audio_data: bytes) -> str:
        """识别说话人"""
        try:
            if not self.database:
                return "unknown"

            audio_tensor = self._preprocess_audio(audio_data)
            embedding = self._extract_embedding(audio_tensor)

            best_user = "unknown"
            best_score = SPEAKER_SIMILARITY_THRESHOLD

            # 计算与所有注册用户的相似度
            for user_id, user_embeddings in self.database.items():
                for user_embedding in user_embeddings:
                    similarity = cosine_similarity(
                        embedding.reshape(1, -1),
                        user_embedding.reshape(1, -1)
                    )[0][0]

                    if similarity > best_score:
                        best_score = similarity
                        best_user = user_id

            if best_user != "unknown":
                print(f"[Speaker] 识别结果: {best_user} (相似度: {best_score:.3f})")
            else:
                print(f"[Speaker] 未识别到已知用户 (最高相似度: {best_score:.3f})")

            return best_user

        except Exception as e:
            print(f"[Speaker] 识别失败: {e}")
            return "unknown"

    def verify(self, user_id: str, audio_data: bytes) -> Tuple[bool, float]:
        """验证用户身份"""
        try:
            if user_id not in self.database:
                return False, 0.0

            audio_tensor = self._preprocess_audio(audio_data)
            embedding = self._extract_embedding(audio_tensor)

            # 计算与该用户所有样本的平均相似度
            similarities = []
            for user_embedding in self.database[user_id]:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    user_embedding.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)

            avg_similarity = np.mean(similarities)
            is_verified = avg_similarity > SPEAKER_SIMILARITY_THRESHOLD

            print(f"[Speaker] 用户验证: {user_id}, 相似度: {avg_similarity:.3f}, 结果: {'通过' if is_verified else '失败'}")

            return is_verified, avg_similarity

        except Exception as e:
            print(f"[Speaker] 验证失败: {e}")
            return False, 0.0

    def get_user_list(self) -> List[str]:
        """获取已注册用户列表"""
        return list(self.database.keys())

    def remove_user(self, user_id: str) -> bool:
        """删除用户"""
        if user_id in self.database:
            del self.database[user_id]
            self._save_database()
            print(f"[Speaker] 用户{user_id}已删除")
            return True
        return False

    def get_user_count(self, user_id: str) -> int:
        """获取用户的注册样本数量"""
        return len(self.database.get(user_id, []))

    def clear_database(self) -> bool:
        """清空所有用户数据"""
        try:
            self.database = {}
            if os.path.exists(SPEAKER_DATABASE_PATH):
                os.remove(SPEAKER_DATABASE_PATH)
            print("[Speaker] 数据库已清空")
            return True
        except Exception as e:
            print(f"[Speaker] 清空数据库失败: {e}")
            return False


# 保持向后兼容的接口
class SpeakerRecognizer(ECAPATDNNRecognizer):
    """
    声纹识别模块 - 兼容旧接口
    """
    pass


def demo_speaker_recognition():
    """演示声纹识别功能"""
    print("=== 声纹识别功能演示 ===")

    recognizer = ECAPATDNNRecognizer()

    # 生成两段不同的测试音频（模拟不同说话人）
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 说话人1：440Hz主频
    speech1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    speech1 += 0.3 * np.sin(2 * np.pi * 880 * t)  # 谐波
    noise1 = 0.1 * np.random.normal(0, 1, len(t))
    audio1 = speech1 + noise1

    # 说话人2：550Hz主频（不同的声纹特征）
    speech2 = 0.5 * np.sin(2 * np.pi * 550 * t)
    speech2 += 0.3 * np.sin(2 * np.pi * 1100 * t)  # 谐波
    noise2 = 0.1 * np.random.normal(0, 1, len(t))
    audio2 = speech2 + noise2

    # 转换为bytes
    audio1_bytes = (audio1 * 32767).astype(np.int16).tobytes()
    audio2_bytes = (audio2 * 32767).astype(np.int16).tobytes()

    print("1. 初始状态识别测试：")
    result1 = recognizer.identify(audio1_bytes)
    result2 = recognizer.identify(audio2_bytes)
    print(f"   音频1识别结果: {result1}")
    print(f"   音频2识别结果: {result2}")

    print("\n2. 用户注册测试：")
    success1 = recognizer.enroll_user("user001", audio1_bytes)
    success2 = recognizer.enroll_user("user002", audio2_bytes)
    print(f"   用户001注册结果: {success1}")
    print(f"   用户002注册结果: {success2}")

    print("\n3. 注册后识别测试：")
    result1_after = recognizer.identify(audio1_bytes)
    result2_after = recognizer.identify(audio2_bytes)
    print(f"   音频1识别结果: {result1_after}")
    print(f"   音频2识别结果: {result2_after}")

    print("\n4. 用户验证测试：")
    verify1 = recognizer.verify("user001", audio1_bytes)
    verify2 = recognizer.verify("user002", audio2_bytes)
    print(f"   用户001验证结果: {verify1}")
    print(f"   用户002验证结果: {verify2}")

    print("\n5. 数据库统计：")
    users = recognizer.get_user_list()
    print(f"   注册用户数量: {len(users)}")
    print(f"   用户列表: {users}")
    for user in users:
        count = recognizer.get_user_count(user)
        print(f"   用户{user}的样本数: {count}")

    print("\n声纹识别演示完成")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_speaker_recognition()
    else:
        # 简单测试
        recognizer = ECAPATDNNRecognizer()
        print("声纹识别模块测试完成")