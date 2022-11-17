from setuptools import setup

def _requires_from_file(filename):
    with open(filename) as f:
        ls = f.read().splitlines()
    return list(filter(lambda x : not (len(x) == 0 or x.startswith('#')), ls))

setup(
    name = "vgg16-for-glide",
    install_requires = _requires_from_file('requirements.txt')
)