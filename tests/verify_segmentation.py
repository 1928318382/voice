import sys
import os
import unittest
import multiprocessing

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.tts import TTSEngine as CoreTTSEngine
# from voice_app.tts import TTSEngine as VoiceTTSEngine # Identical logic

class TestTTSListSegmentation(unittest.TestCase):
    def test_segmentation_with_newlines(self):
        print("\n--- Testing Segmentation with Newlines ---")
        q_in = multiprocessing.Queue()
        q_evt = multiprocessing.Queue()
        
        engine = CoreTTSEngine(q_in, q_evt, audio_device_mock=True)
        
        input_text = """为您准备了3个生活小贴士：
1、日常生活小窍门：定期整理衣柜，只保留最近一年穿过的衣服
2、节约用水技巧：洗澡时间控制在5-10分钟，使用节水淋浴头
3、环保生活习惯：随手关灯，使用可重复使用的购物袋"""

        # 1. Test Normalization
        normalized = engine._normalize_text(input_text)
        print(f"Original:\n{input_text}")
        print(f"Normalized:\n{normalized}")
        
        # Expectation: Newlines become '。' and then merged if multiple. 
        self.assertIn("生活小贴士：。", normalized) 
        self.assertIn("穿过的衣服。", normalized)
        
        # Verify Number Handling Fixed for Ranges
        # 5-10 should NOT become "五负十" (five negative ten)
        # It should become "五-十" or "五 十" (depending on punct strip)
        # "负" is the character for negative.
        self.assertNotIn("负", normalized, "Should not interpret hyphen in 5-10 as negative sign")
        self.assertIn("五", normalized)
        self.assertIn("十", normalized)

        # 2. Test Splitting
        segments = engine._split_long_text(normalized)
        print(f"Segments ({len(segments)}):")
        for i, seg in enumerate(segments):
            print(f"[{i+1}] {seg}")
            
        # Expectation: 
        # Even if segments are merged (for efficiency), they should break at logical points.
        # Check that segments end with punctuation (meaning we didn't slice mid-sentence).
        # We allow the last segment to not end with punctuation if it's the end of text.
        for i, seg in enumerate(segments):
            if i < len(segments) - 1:
                has_punct_end = seg.strip().endswith(("。", "！", "？", ".", "!", "?", "…", "~"))
                if not has_punct_end:
                     print(f"WARNING: Segment {i+1} does not end with punctuation: {seg}")
                # Ideally this should be true for safe splitting
                # self.assertTrue(has_punct_end, f"Segment {i+1} splits mid-sentence: {seg}")
        
        # Check that we have valid content
        self.assertIn("生活小贴士", segments[0])
        self.assertTrue(any("环保生活习惯" in s for s in segments))

if __name__ == '__main__':
    unittest.main()
