VERSION = (0, 0, 9)

__version__ = ".".join(map(str, VERSION))
short_version = __version__


import sys

msg = "Open-retrievals is only compatible with Python 3.7 and newer, please consider a newer version."

if sys.version_info < (3, 6):
    raise ImportError(msg)
