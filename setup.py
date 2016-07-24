try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup
	
config = {
    'description': 'Human motion prediction',
    'author': 'Antonio A. Matta-Gomez',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'amattag@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'numpy', 'scipy'],
    'packages': ['motionPred'],
    'scripts': [],
    'name': 'motionPredictor'
}

setup(**config)
