from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fractal-generator",
    version="1.0.0",
    author="Fractal Generator",
    description="High-quality fractal generation with advanced rendering capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "web": ["streamlit>=1.0.0", "flask>=2.0.0"],
        "video": ["imageio[ffmpeg]>=2.10.0"],
        "precision": ["mpmath>=1.2.0"],
    },
    entry_points={
        "console_scripts": [
            "fractal-gen=fractal_generator.cli.main:main",
        ],
    },
)
