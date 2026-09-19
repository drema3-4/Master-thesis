def h0(k: float, x: float) -> float:
    k = float(k)
    x = float(x)

    return k * x

def h1(k: float, x: float, alpha: float) -> float:
    k = float(k)
    x = float(x)
    alpha = float(alpha)

    return k * x + alpha * x**3