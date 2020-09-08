from setuptools import setup
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

name = 'morphology'
version = '0.0.0'
release = '0.0.0'
setup(
    name='morphology',
    version='0.0.0',
    author='Toth Technology',
    author_email='toth-tech@hillyer.me',
    description='Morphology Implementation',
    long_description='TBD',
    long_description_content_type='text/markdown',
    url='https://github.com/neshdev/pypy_test',
    project_urls={
        'Github': 'https://github.com/neshdev/pypy_test'
    },
    packages=['morphology'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    install_requires=[
        "numpy",
        "scikit-image",
        "pyvista",
    ],
    python_requires='>=3.8',
    keywords='TBD',
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs/source'),
            'config_dir': ('setup.py', 'docs/source'),
            'build_dir' : ('setup.py', 'docs/build')
            
        }
    },
)