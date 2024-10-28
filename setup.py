import io
import os
from setuptools import (
    setup,
    find_packages,
)


def read(
    fname,
):
    with io.open(
        os.path.join(
            os.path.dirname(__file__),
            fname,
        ),
        encoding="utf-8",
    ) as f:
        return f.read()


setup(
    name="morered",
    version="1.0.0",
    author="Khaled Kahouli",
    url="https://github.com/twoPaiSquared/MoreRed",
    packages=find_packages("src"),
    package_dir={"": "src"},
    scripts=[
        "src/scripts/mrdtrain",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"": ["configs/**/*.yaml"]},
    install_requires=["schnetpack==2.0.3", "progressbar"],
    license="MIT",
    description="MoreRed - Molecular Relaxation with Reverse Diffusion.",
    long_description="""
        MoreRed is an extension of SchNetPack that implements
        diffusion models for molecular structures. Its functionality
        encompasses geometry optimization and the generation of novel
        structures based on chemical compositions. """,
)
