from typing import Dict
from genesys.schemas import Response


class BaseVerifier:
    """
    Base class for all verifiers.
    """

    max_parallel: int = 5
    timeout: float = None  # None means no timeout

    def verify(self, result: Response) -> Dict:
        """Perform the synchronous verification given a single result.

        Subclasses should override this to implement the actual check.
        """
        raise NotImplementedError("Subclasses must implement verify().")
    
    def terminate():
        pass
