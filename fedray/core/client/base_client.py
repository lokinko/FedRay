from abc import abstractmethod


class BaseClientActor:
    def __init__(self, dataset=None) -> None:
        self._dataset = dataset

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def eval():
        pass
