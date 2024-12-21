from setuptools import setup, find_packages


setup(
    name="llms",
    version="0.1.0",
    description="Arabic VQA",
    author="Mahmood Anaam",
    author_email="eng.mahmood.anaam@gmail.com",
    url="https://github.com/Mahmood-Anaam/llms",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=[
        "transformers==4.46.3",
        "torch==2.5.1+cu121",
        "datasets==3.2.0",
        "numpy==1.26.4",
        "google-generativeai>=0.7.2"
        
        ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)


