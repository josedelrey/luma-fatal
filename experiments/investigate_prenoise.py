"""Compare noise before Fattal using the existing still/synthetic/video runner.

The configured case accepts --noise-rms, --noise-kind, --noise-cleanup,
--noise-unmasked and --noise-seed. All other options match investigate_cleanup.py.
"""

import argparse

import investigate_cleanup as study
from luma_fatal.cleanup import SourceCleanup
from prenoise import SourceNoise


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--noise-rms", type=float, default=0.5)
    parser.add_argument("--noise-kind", choices=("highpass", "fbm"), default="highpass")
    parser.add_argument("--noise-cleanup", type=float, default=0)
    parser.add_argument("--noise-unmasked", action="store_true")
    parser.add_argument("--noise-seed", type=int, default=20261017)
    options, remaining = parser.parse_known_args()
    settings = dict(rms=options.noise_rms, kind=options.noise_kind,
                    cleanup=options.noise_cleanup, masked=not options.noise_unmasked,
                    seed=options.noise_seed)
    SourceNoise(**settings)
    study.CASES.clear()
    study.CASES.update({
        "off": lambda: None,
        "current075": lambda: SourceCleanup(0.75),
        "highpass035": lambda: SourceNoise(0.35),
        "highpass075": lambda: SourceNoise(0.75),
        "fbm075": lambda: SourceNoise(0.75, kind="fbm"),
        "clean075-highpass035": lambda: SourceNoise(0.35, cleanup=0.75),
        "clean075-highpass075": lambda: SourceNoise(0.75, cleanup=0.75),
        "unmasked075": lambda: SourceNoise(0.75, masked=False),
        "configured": lambda: SourceNoise(**settings),
    })
    study.__doc__ = __doc__
    study.main(remaining, extra_record={
        "configured_noise": settings,
        "noise_policy": "Fixed working-frame coordinates, no motion compensation",
        "case_policy": "Cases explicitly replace configured flat cleanup",
    })


if __name__ == "__main__":
    main()
