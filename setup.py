import subprocess
import setuptools

# install inplace_abn 
# subprocess.call(["pip", "install", "git+https://github.com/mapillary/inplace_abn.git@v1.0.11"])
subprocess.call(['python', 'magic_vnet/blocks/inplace_abn/setup.py', 'install'])

setuptools.setup(
    # Meta-data
    name="magic-vnet",
    author="HsuXu",
    version='0.0.2',
    author_email="hsuxu820@gmail.com",
    description="VNet family implemented in PyTorch",
    url="https://github.com/Hsuxu/Magic-VNet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    # Version
    use_scm_version={"root": ".", "relative_to": __file__, "write_to": "magic_vnet/_version.py"},

    # Requirements
    python_requires=">=3, <4",

    # Package description
    packages=["magic_vnet"],

)
