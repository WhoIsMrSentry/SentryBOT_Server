from abc import ABC, abstractmethod
from typing import Optional, Any

class BaseService(ABC):
    @abstractmethod
    def initialize(self):
        """Load models or resources."""
        pass

    @abstractmethod
    def health_check(self) -> dict:
        """Return health status."""
        pass
