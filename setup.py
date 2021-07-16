from setuptools import setup, find_namespace_packages

setup(name='kits21',
      packages=find_namespace_packages(),
      version='2.2.2',
      description='',
      zip_safe=False,
      install_requires=[
            'batchgenerators',
            'numpy',
            'SimpleITK',
            'medpy',
            'nibabel',
            'pillow',
            'opencv-python',
            'torch',
            'scipy',
            'scikit-image',
            'requests',
            'argparse'
      ])
