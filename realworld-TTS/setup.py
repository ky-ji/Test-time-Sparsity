"""
TTS Accelerator - 


    cd realworld-TTS
    pip install -e .
    

    from tts_accelerator import TTSPolicyWrapper, load_pruner
"""
from setuptools import setup, find_packages

setup(
    name="tts-accelerator",
    version="0.1.0",
    description="TTS Accelerator for Diffusion Policy",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            # NOTE: run_dp_with_tts.py moved into tts_accelerator/scripts/
            "run-dp-tts=tts_accelerator.scripts.run_dp_with_tts:main",
        ],
    },
)
