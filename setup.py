from __future__ import annotations

from setuptools import find_packages, setup


setup(
    name="openharmony-anns-test",
    version="0.1.0",
    description="Generic filtered ANN acceptance test harness",
    packages=find_packages(include=["anns_acceptance", "anns_acceptance.*"]),
    python_requires=">=3.10",
    install_requires=[
        "pytest>=8.0",
        "pydantic>=2.7",
        "psutil>=5.9",
        "typer>=0.12",
        "rich>=13.0",
        "PyYAML>=6.0",
        "numpy>=1.26",
    ],
    entry_points={
        "console_scripts": ["anns-acceptance=anns_acceptance.cli:main"],
        "pytest11": ["anns_acceptance=anns_acceptance.pytest_plugin"],
    },
)
