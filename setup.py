from setuptools import setup, find_packages

setup(
    name="aravqa",
    version="0.1.0",
    description="arabic-visual-question-answering",
    author="Mahmood Anaam",
    author_email="eng.mahmood.anaam@gmail.com",
    url="https://github.com/Mahmood-Anaam/arabic-visual-question-answering.git",
    license="MIT",
    packages=find_packages(exclude=["notebooks", "assets", "scripts", "tests"]),
    install_requires=[
        "vinvl_bert @ git+https://github.com/Mahmood-Anaam/vinvl_bert.git",
        "Violet @ git+https://github.com/Mahmood-Anaam/Violet.git",
        "tqdm==4.67.1",
        "google-generativeai==0.8.2",
        "huggingface-hub==0.27.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.47.1",
        "evaluate==0.4.3",
        "datasets==3.2.0",
        "arabert==1.0.1",
       
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
