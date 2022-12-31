# from .clip import *
try:
    from .eva_clip import *
except ModuleNotFoundError:
    from BLIP2.eva_clip.eva_clip import *
