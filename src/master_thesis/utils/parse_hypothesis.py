def parse_hypothesis(content: str) -> str:
    normalized = content.strip().upper()

    if normalized == "H0":
        return "H0"

    if normalized == "H1":
        return "H1"

    return "invalid"