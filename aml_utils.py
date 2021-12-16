LAMBDA: float = None

def set_lambda(value: float = None):
    global LAMBDA
    LAMBDA = float(value)

def get_lambda() -> float:
    return LAMBDA