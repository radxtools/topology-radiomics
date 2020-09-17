import os
from setuptools import setup
from sphinx.setup_command import BuildDoc
from datetime import date
import subprocess
from pathlib import Path
cmdclass = {'build_sphinx': BuildDoc}


def compute_version():
    # Will not be using date versioning because github run number keeps incrementing by 1
    # Leaving this code here incase we want to revisit
    now = date.today()
    yy = now.year
    mm = now.month
    dd = now.day
    version = f"{yy}.{mm}.{dd}"
    version = (Path(__file__).parent / "version.txt").open().read()

    if os.environ.get('GITHUB_RUN_NUMBER'):
        run_number = os.environ.get('GITHUB_RUN_NUMBER')
        print("Using github workflow")
        ref = os.environ.get('GITHUB_REF')
        if ref:
            ref = ref.strip()
            branch_name = ref.split("/")[-1]
            if branch_name != "master":
                version = f"{version}.dev{run_number}"
    else:
        print("Using shell to get branch name")
        process = subprocess.run(
            ['git', 'branch', '--show-current'], capture_output=True)
        branch = process.stdout.decode('utf-8').strip()
        if branch != 'master':
            version = f"{version}.dev1"
    return version


name = 'topology_radiomics'
version = compute_version()
release = version
setup(
    name=name,
    version=version,
    author='Toth Technology',
    author_email='toth-tech@hillyer.me',
    description='Topology Radiomics by INVENT and BRIC lab',
    long_description='TBD',
    long_description_content_type='text/markdown',
    url='https://github.com/Toth-Technology/bric-morphology',
    project_urls={
        'Github': 'https://github.com/Toth-Technology/bric-morphology'
    },
    packages=['topology_radiomics'],
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
        "pandas",
        "sphinx",
    ],
    python_requires='>=3.8',
    keywords='Topology Radiomics',
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs/source'),
            'config_dir': ('setup.py', 'docs/source'),
            'build_dir': ('setup.py', 'docs/build')
        }
    },
)
