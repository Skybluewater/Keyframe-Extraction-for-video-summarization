from abc import ABC, abstractmethod

class BaseExtraction(ABC):
    @abstractmethod
    def clustering(features, **kwargs):
        pass
