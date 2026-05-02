import sys
try:
    import robobase
    print(robobase.__file__)
except ImportError:
    print("Could not import robobase")
