import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='landsurvey',
    version='0.0.1',
    author='wslerry',
    description='Semantic segmentation on aerial and satellite imagery.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wslerry/robosat',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'flask~=1.0',
        'geojson~=2.4',
        'matplotlib~=3.1',
        'mercantile~=1.0',
        'numpy~=1.16',
        'opencv-contrib-python-headless~=4.0',
        'osmium==2.15.2',
        'pillow~=6.0',
        'pyproj~=2.1',
        'rasterio~=1.0',
        'requests~=2.22',
        'rtree~=0.8',
        'scipy~=1.3',
        'shapely~=1.6',
        'supermercado~=0.0.5',
        'toml~=0.10',
        'torch~=1.1',
        'torchvision~=0.3',
        'tqdm~=4.32',
    ],
    entry_points = {
        'console_scripts': [
            'landsurvey=landsurvey.tools.__main__:main'
        ]
    }
)
