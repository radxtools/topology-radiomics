from setuptools import setup

setup(
    name='morphology',
    version='0.0.0',
    author='Toth Technology',
    author_email='toth-tech@hillyer.me',
    description='Morphology Implementation',
    long_description='TBD',
    long_description_content_type='text/markdown',
    url='https://github.com/Toth-Technology/bric-morphologys',
    project_urls={
        'Docker Examples': 'TBD',
        'Docker Module': 'TBD',
        'Github': 'https://github.com/Toth-Technology/bric-morphology'
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
)