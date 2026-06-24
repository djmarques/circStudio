"""circstudio.preprocessing — signal preprocessing utilities.

Currently provides automatic non-wear detection for actigraphy count data.

Algorithms
----------
detect_nonwear_troiano
    Troiano et al. (2008) NHANES algorithm: flags runs of ≥60 consecutive
    minutes of zero activity (with limited spike tolerance).

detect_nonwear_choi
    Choi et al. (2011) improved NHANES algorithm: same spike tolerance as
    Troiano but also requires that 30-minute windows on either side of any
    allowed spike are themselves zero, reducing false positives during sleep.
"""

from .nonwear import detect_nonwear_choi, detect_nonwear_troiano

__all__ = [
    "detect_nonwear_troiano",
    "detect_nonwear_choi",
]
