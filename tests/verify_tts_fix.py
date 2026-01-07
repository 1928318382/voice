import sys
import os
import unittest
import multiprocessing

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.tts import _number_to_chinese, TTSEngine as CoreTTSEngine
from voice_app.tts import _number_to_chinese as voice_number_to_chinese, TTSEngine as VoiceTTSEngine

class TestTTSFixes(unittest.TestCase):
    def test_number_conversion(self):
        test_cases = [
            ("123", "一百二十三"),
            ("0", "零"),
            ("10", "十"),
            ("11", "十一"),
            ("100", "一百"),
            ("101", "一百零一"),
            ("1000", "一千"),
            ("1001", "一千零一"),
            ("10000", "一万"),
            ("12345678", "一千二百三十四万五千六百七十八"), # 8 digits -> full read
            ("123456789", "一二三四五六七八九"), # >8 digits -> digit-by-digit
            ("3.14", "三点一四"),
            ("-5", "负五"),
            ("13800138000", "一三八零零一三八零零零"), # Phone number style
        ]
        
        print("\nTesting app.core.tts number conversion:")
        for input_str, expected in test_cases:
            result = _number_to_chinese(input_str)
            print(f"  {input_str} -> {result}")
            self.assertEqual(result, expected)

        print("\nTesting voice_app.tts number conversion:")
        for input_str, expected in test_cases:
            result = voice_number_to_chinese(input_str)
            self.assertEqual(result, expected)

    def test_normalize_text(self):
        # Mock queues for init
        q_in = multiprocessing.Queue()
        q_evt = multiprocessing.Queue()
        
        engine = CoreTTSEngine(q_in, q_evt, audio_device_mock=True)
        
        text = "温度是25.5度"
        normalized = engine._normalize_text(text)
        print(f"\nNormalized: {text} -> {normalized}")
        self.assertIn("二十五点五", normalized)
        
        text_with_eng = "Hello 123 World"
        normalized = engine._normalize_text(text_with_eng)
        print(f"Normalized: {text_with_eng} -> {normalized}")
        self.assertNotIn("Hello", normalized) # English should be removed
        self.assertIn("一百二十三", normalized) # Number should be converted

    def test_split_long_text(self):
        q_in = multiprocessing.Queue()
        q_evt = multiprocessing.Queue()
        engine = CoreTTSEngine(q_in, q_evt, audio_device_mock=True)
        
        # Test 1: Simple long sentence with punctuation
        long_sentence = "这是一个非常长的句子，它应该被切分成多个部分。这里是第二部分，希望它能工作。"
        # Force MAX_SENTENCE_LENGTH to be small for testing
        import app.core.tts
        original_max = app.core.tts.MAX_SENTENCE_LENGTH
        app.core.tts.MAX_SENTENCE_LENGTH = 10
        
        # Reload method to use new constant if it was bound (it's not, it reads module global usually)
        # But wait, MAX_SENTENCE_LENGTH is imported/defined in the module.
        # The class method uses the global MAX_SENTENCE_LENGTH from the module where it's defined.
        
        try:
            segments = engine._split_long_text(long_sentence)
            print(f"\nSplit result (limit 10): {segments}")
            for seg in segments:
                self.assertLessEqual(len(seg), 15) # Allow some buffer, logic tries to keep punctuation
        finally:
            app.core.tts.MAX_SENTENCE_LENGTH = original_max

if __name__ == '__main__':
    unittest.main()
