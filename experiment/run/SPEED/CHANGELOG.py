MODEL_VERSION = "v1.0.0"
CHANGELOG = """
v1.0.0:
    1. SPEED
"""



def CHECK_VERSION(v1, v2, level=2):
    v1 = v1.split('.')[:level]
    v2 = v2.split('.')[:level]
    return v1 == v2
