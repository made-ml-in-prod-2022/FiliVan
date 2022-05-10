from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="ml project",
    author="FiliVan",
    entry_points={
        "console_scripts": [
            "ml_project_train = ml_example.train:train_command",
            "ml_project_predict = ml_example.predict:predict_command",
        ]
    },
    install_requires=required,
    license="MIT",
)
