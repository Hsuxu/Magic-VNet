import setuptools

setuptools.setup(
    # Meta-data
    name="magic_vnet",
    author="HsuXu",
    version='0.0.2',
    author_email="hsuxu820@gmail.com",
    description="VNet family implemented by PyTorch",
    url="https://github.com/Hsuxu/Magic-VNet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    # Versioning
    use_scm_version={"root": ".", "relative_to": __file__, "write_to": "magic_vnet/_version.py"},

    # Requirements
    python_requires=">=3, <4",

    # Package description
    packages=["magic_vnet"],

    # install_requires=['inplace_abn>=1.0.11'],
    # dependency_links=[
    #     "git+https://github.com/mapillary/inplace_abn.git@v1.0.11"
    # ]
)
