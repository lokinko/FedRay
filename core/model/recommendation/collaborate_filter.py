import copy
import torch

class CollaborativeFilterModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_items = args['num_items']
        self.item_hidden_dim = args['item_hidden_dim']

        self.item_personality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)
        self.item_commonality = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.item_hidden_dim)

        self.user_embedding = torch.nn.Linear(in_features=self.item_hidden_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def setItemCommonality(self, item_commonality):
        self.item_commonality = copy.deepcopy(item_commonality)
        self.item_commonality.freeze = True

    def forward(self, item_indices):
        item_personality = self.item_personality(item_indices)
        item_commonality = self.item_commonality(item_indices)

        logits = self.affine_output(item_personality + item_commonality)
        rating = self.logistic(logits)

        return rating, item_personality, item_commonality
