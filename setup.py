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
   
    # install_requires=[
    #     "vinvl@git+https://github.com/Mahmood-Anaam/vinvl.git",
    #     "vinvl_bert@git+https://github.com/Mahmood-Anaam/vinvl_bert.git",
    #     "Violet@git+https://github.com/Mahmood-Anaam/Violet.git",
    #     "yacs",
    #     "transformers",
    #     "google-generativeai>=0.8.2",
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
