import ray

from fedray.core.client import BaseClientActor


@ray.remote
class FederatedClient(BaseClientActor):
    def __init__(self, model, optimizer, dataset=None):
        super().__init__(dataset=dataset)
        self._model = model
        self._optimizer = optimizer

    def train():
        pass

    def eval():
        pass
