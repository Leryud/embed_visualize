from datetime import datetime
from typing import List, Dict

def datetime_string_to_unix_timestamp(date_string, format_string) -> int:
    """
    Convert a datetime string to a Unix timestamp (integer).

    Args:
        date_string (str): The datetime string to convert.
        format_string (str): The format of the datetime string.

    Returns:
        int: The Unix timestamp.
    """
    # Parse the datetime string
    dt = datetime.strptime(date_string, format_string)

    # Convert to Unix timestamp (float)
    timestamp_float = dt.timestamp()

    # Convert to integer
    timestamp_int = int(timestamp_float)

    return timestamp_int

def filter_by_date(corpus: List[Dict[str, str]], filter_key: str = "date"):
    min_date = 1792447200
    max_date = 1792504403
    format_string = "%Y-%m-%dT%H:%M:%SZ"

    filtered_corpus = [
        item for item in corpus
        if min_date
        <= datetime_string_to_unix_timestamp(item.get(filter_key, 0), format_string)
        <= max_date
    ]

    return filtered_corpus
