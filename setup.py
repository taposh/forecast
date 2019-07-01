from setuptools import find_packages
from setuptools import setup

required = [
      "forecast",
      "pandas",
      "python-dateutil",
      "numpy",
      "sklearn",
      "torch"
]

setup(name='kpforecast',
      version='0.0.2',
      description='The forecast package used by kp insight',
      url='https://github.com/taposh/forecast.git',
      author='KP-Insight',
      author_email='taposh.d.roy@kp.org',
      license='MIT',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      install_requires=required,
      zip_safe=False)
