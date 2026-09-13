"""Hardware detection and conservative runtime profiles for Apple Silicon."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
import subprocess


GIB = 1024**3


@dataclass(frozen=True)
class HardwareInfo:
    architecture: str
    chip: str
    memory_gib: float


@dataclass(frozen=True)
class RuntimeSettings:
    profile: str
    memory_limit_gib: float
    dense_cache_gib: float
    mlx_cache_mib: int


PROFILES = {
    "low-memory": RuntimeSettings("low-memory", 8, 4, 256),
    "apple-32gb": RuntimeSettings("apple-32gb", 18, 10, 384),
    "apple-48gb": RuntimeSettings("apple-48gb", 28, 16, 512),
    "m5-max-48gb": RuntimeSettings("m5-max-48gb", 28, 16, 512),
    "apple-64gb": RuntimeSettings("apple-64gb", 38, 24, 768),
    "apple-128gb": RuntimeSettings("apple-128gb", 76, 52, 1024),
}


def _sysctl(name: str) -> str:
    try:
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def detect_hardware() -> HardwareInfo:
    memory = _sysctl("hw.memsize")
    chip = _sysctl("machdep.cpu.brand_string")
    return HardwareInfo(
        architecture=platform.machine(),
        chip=chip or "Unknown Apple Silicon",
        memory_gib=int(memory) / GIB if memory.isdigit() else 0.0,
    )


def auto_profile(info: HardwareInfo) -> str:
    if info.memory_gib < 24:
        return "low-memory"
    if info.memory_gib < 44:
        return "apple-32gb"
    if info.memory_gib < 58:
        return "m5-max-48gb" if "M5 Max" in info.chip else "apple-48gb"
    if info.memory_gib < 96:
        return "apple-64gb"
    return "apple-128gb"


def resolve_settings(
    profile_name: str = "auto",
    *,
    info: HardwareInfo | None = None,
    memory_limit_gib: float | None = None,
    dense_cache_gib: float | None = None,
    mlx_cache_mib: int | None = None,
) -> RuntimeSettings:
    info = info or detect_hardware()
    selected = auto_profile(info) if profile_name == "auto" else profile_name
    if selected not in PROFILES:
        raise ValueError(f"Unknown runtime profile: {selected}")
    base = PROFILES[selected]
    result = RuntimeSettings(
        profile=selected,
        memory_limit_gib=memory_limit_gib if memory_limit_gib is not None else base.memory_limit_gib,
        dense_cache_gib=dense_cache_gib if dense_cache_gib is not None else base.dense_cache_gib,
        mlx_cache_mib=mlx_cache_mib if mlx_cache_mib is not None else base.mlx_cache_mib,
    )
    if result.memory_limit_gib <= 2:
        raise ValueError("memory limit must be greater than 2 GiB")
    if not 0 <= result.dense_cache_gib <= result.memory_limit_gib - 2:
        raise ValueError("dense cache must leave at least 2 GiB inside the MLX memory limit")
    if result.mlx_cache_mib < 0:
        raise ValueError("MLX cache limit cannot be negative")
    if info.memory_gib and result.memory_limit_gib > info.memory_gib * 0.8:
        raise ValueError("memory limit exceeds 80% of physical memory")
    return result


def main() -> None:
    info = detect_hardware()
    settings = resolve_settings(info=info)
    print(json.dumps({"hardware": asdict(info), "runtime": asdict(settings)}, indent=2))


if __name__ == "__main__":
    main()
