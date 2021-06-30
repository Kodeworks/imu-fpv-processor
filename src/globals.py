# these are overridden in numeric_service, so it's safe to change them as long as type is the same
import platform

numberOfFloats: str
stats: list
statFuncs: list
fibs: list
cols: int
rows: int


def isLinux():
    return 'Linux' == platform.system()
