from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand
import sys

with open("README.md", "r") as readme:
    long_description = readme.read()


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        sys.exit(pytest.main(self.test_args))


class PyTestSkipSlow(PyTest):
    def finalize_options(self):
        super(PyTestSkipSlow, self).finalize_options()
        self.test_args.append("-m not slow")


setup(
    name="stockfish",
    author="Ilya Zhelyabuzhsky",
    author_email="zhelyabuzhsky@icloud.com",
    version="3.29.0",
    license="MIT",
    keywords="chess stockfish",
    python_requires=">=3.7",
    url="https://github.com/zhelyabuzhsky/stockfish",
    description="Wraps the open-source Stockfish chess engine for easy integration into python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["stockfish", "stockfish.*"]),
    install_requires=[],
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov"],
    classifiers=[
        "Programming Language :: Python",
        "Natural Language :: English",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    cmdclass={"test": PyTest, "skip_slow_tests": PyTestSkipSlow},
)
