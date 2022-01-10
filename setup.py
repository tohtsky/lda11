import os
import sys
from typing import Any, Dict, List

import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "0.3.0.0"
install_requires = ["numpy >= 1.22", "tqdm", "scipy>=1.0.0"]
setup_requires = ["pybind11>=2.5", "requests", "setuptools_scm"]


eigen_include_dir = os.environ.get("EIGEN3_INCLUDE_DIR", None)
if eigen_include_dir is None:
    install_requires.append("requests")

TEST_BUILD = os.environ.get("TEST_BUILD", None) is not None


class get_eigen_include(object):
    EIGEN3_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip"
    EIGEN3_DIRNAME = "eigen-3.3.7"

    def __str__(self) -> str:
        if eigen_include_dir is not None:
            return eigen_include_dir

        basedir = os.path.dirname(__file__)
        target_dir = os.path.join(basedir, self.EIGEN3_DIRNAME)
        print(target_dir)
        if os.path.exists(target_dir):
            return target_dir

        download_target_dir = os.path.join(basedir, "eigen3.zip")
        import zipfile

        import requests

        response = requests.get(self.EIGEN3_URL, stream=True)
        with open(download_target_dir, "wb") as ofs:
            for chunk in response.iter_content(chunk_size=1024):
                ofs.write(chunk)

        with zipfile.ZipFile(download_target_dir) as ifs:
            ifs.extractall()

        return target_dir


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""

    def __init__(self, user=False):
        self.user = user

    def __str__(self) -> str:
        import pybind11

        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        "lda11._lda",
        [
            "src/wrapper.cpp",
            "src/predictor.cpp",
            "src/trainer_base.cpp",
            "src/trainer.cpp",
            "src/child_worker.cpp",
            "src/labelled_lda.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            get_eigen_include(),
        ],
        language="c++",
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname) -> bool:
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler) -> str:
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    if TEST_BUILD:
        c_opts: Dict[str, List[str]] = {
            "msvc": ["/EHsc"],
            "unix": ["-O0", "-coverage", "-g"],
        }
        l_opts: Dict[str, List[str]] = {
            "msvc": [],
            "unix": ["-coverage"],
        }
    else:
        c_opts = {
            "msvc": ["/EHsc"],
            "unix": [],
        }
        l_opts = {
            "msvc": [],
            "unix": [],
        }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self) -> None:
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


def local_scheme(version: Any) -> str:
    return ""


setup(
    name="lda11",
    use_scm_version={"local_scheme": local_scheme},
    version=__version__,
    author="Tomoki Ohtsuki",
    url="https://github.com/tohtsky/lda11",
    author_email="tomoki.ohtsuki129@gmail.com",
    description="Yet another CGS sampler for Latent Dirichlet Allocation.",
    long_description="",
    ext_modules=ext_modules,
    install_requires=install_requires,
    setup_requires=setup_requires,
    cmdclass={"build_ext": BuildExt},
    packages=["lda11"],
    zip_safe=False,
)
