import re

def sanitize_key(value: str) -> str:
    """
    Converts a human-readable category name (e.g. 'IT-Management, Consulting (und Auditing)')
    into a Qdrant-safe key: lowercase, no spaces, no commas, only [a-z0-9_].

    Example:
        'IT-Management, Consulting (und Auditing)' → 'it_management_consulting_und_auditing'
    """
    if not value:
        return ""
    value = value.lower()
    value = value.replace("&", "and")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")
