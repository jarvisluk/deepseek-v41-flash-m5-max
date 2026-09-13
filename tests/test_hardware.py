import unittest

from deepseek_v41_mlx.hardware import HardwareInfo, auto_profile, resolve_settings


class HardwareProfileTests(unittest.TestCase):
    def test_auto_profiles(self):
        self.assertEqual(auto_profile(HardwareInfo("arm64", "Apple M1", 16)), "low-memory")
        self.assertEqual(auto_profile(HardwareInfo("arm64", "Apple M4 Pro", 32)), "apple-32gb")
        self.assertEqual(auto_profile(HardwareInfo("arm64", "Apple M5 Max", 48)), "m5-max-48gb")
        self.assertEqual(auto_profile(HardwareInfo("arm64", "Apple M4 Max", 48)), "apple-48gb")
        self.assertEqual(auto_profile(HardwareInfo("arm64", "Apple M3 Ultra", 64)), "apple-64gb")
        self.assertEqual(auto_profile(HardwareInfo("arm64", "Apple M3 Ultra", 128)), "apple-128gb")

    def test_m5_max_profile_is_conservative(self):
        info = HardwareInfo("arm64", "Apple M5 Max", 48)
        settings = resolve_settings(info=info)
        self.assertEqual(settings.profile, "m5-max-48gb")
        self.assertEqual(settings.memory_limit_gib, 28)
        self.assertEqual(settings.dense_cache_gib, 16)
        self.assertLessEqual(settings.memory_limit_gib, info.memory_gib * 0.8)

    def test_overrides_are_validated(self):
        info = HardwareInfo("arm64", "Apple M5 Max", 48)
        settings = resolve_settings(
            "m5-max-48gb", info=info, memory_limit_gib=24, dense_cache_gib=12
        )
        self.assertEqual(settings.memory_limit_gib, 24)
        self.assertEqual(settings.dense_cache_gib, 12)


        with self.assertRaisesRegex(ValueError, "leave at least 2 GiB"):
            resolve_settings(
                "m5-max-48gb", info=info, memory_limit_gib=20, dense_cache_gib=19
            )


if __name__ == "__main__":
    unittest.main()
