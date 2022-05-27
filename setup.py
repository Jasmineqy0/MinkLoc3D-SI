from setuptools import setup

setup(
    name='MinkLoc3D-SI',
    version='1.0.0',
    # package_dir={'': 'inside_out_detection'},
    packages=['datasets', 'eval', 'images', 'layers', 'misc', 'models', 'training'],
    install_requires=[],
    zip_safe=False,
    include_package_data=True
)