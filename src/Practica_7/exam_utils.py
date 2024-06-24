def green_text(text: str) -> str:
    """Returns the text in green color.

    Args:
        text (str): text to color

    Returns:
        str: colored text
    """
    return f"\033[92m{text}\033[00m"


def white_text(text: str) -> str:
    """Returns the text in white color.

    Args:
        text (str): text to color

    Returns:
        str: colored text
    """
    return f"\033[97m{text}\033[00m"


def red_text(text: str) -> str:
    """Returns the text in red color.

    Args:
        text (str): text to color

    Returns:
        str: colored text
    """
    return f"\033[91m{text}\033[00m"
