from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as infd:
    INSTALL_REQUIRES = [x.strip('\n') for x in infd.readlines()]
setup(
    name="resamvo",
    version="0.1",
    author="Zeyu Gao",
    author_email="zygao@stu.pku.edu.cn",
    license='MIT',
    description="Resample data with Voronoi tessellation",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.6',
)