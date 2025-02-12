import hashlib


def sha256sum(filename: str, chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 checksum of a file efficiently.

    Args:
        filename (str): Path to the file.
        chunk_size (int, optional): Size of chunks read at a time. Defaults to 64 KB.

    Returns:
        str: The SHA-256 hash of the file as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    with open(filename, "rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(memoryview(chunk))
    return sha256.hexdigest()
