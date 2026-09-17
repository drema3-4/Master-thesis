def sse(truths: list[float], predicteds: list[float]) -> float:
    sse = 0

    for truth, predicted in zip(truths, predicteds):
        diff = truth - predicted
        sse += diff * diff

    return float(sse)