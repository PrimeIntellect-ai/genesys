from typing import Dict
from genesys.schemas import UnscoredResult


class BaseVerifier:
    """
    Base class for all verifiers.
    """

    max_parallel: int = 5
    timeout: float = None  # None means no timeout

    def verify(self, result: UnscoredResult) -> Dict:
        """Perform the synchronous verification given a single result.

        Subclasses should override this to implement the actual check.
        """
        raise NotImplementedError("Subclasses must implement verify().")
