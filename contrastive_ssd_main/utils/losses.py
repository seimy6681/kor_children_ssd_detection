import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        
        """
        margin: float — the margin that the negative should be farther than the positive
        p: int — the norm degree for pairwise distance (p=2 for Euclidean)
        """
        super(TripletLoss, self).__init__() # calling the parent class constructor #super().__init__()
        # super().__init() # calling nn.Module constructor
        self.margin = margin
        self.p = p # =2 for euclidean distance, =1 for manhattan distance


    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: tensors of shape (batch_size, embedding_dim)
        """
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        pos_distance = F.pairwise_distance(anchor, positive, p=self.p)
        neg_distance = F.pairwise_distance(anchor, negative, p =self.p)
        
        pos_distance_avg = pos_distance.mean()
        neg_distance_avg = neg_distance.mean()
        
        cos_sim_pos = F.cosine_similarity(anchor, positive)
        cos_sim_neg = F.cosine_similarity(anchor, negative)
        
        # Higher is better
        losses = torch.clamp(cos_sim_neg - cos_sim_pos + self.margin, min=0).mean()

        # losses = F.relu(-neg_distance + self.margin)
        # losses = F.relu(pos_distance - neg_distance + self.margin) # sets negative values to 0
        # losses = F.relu(pos_distance)
        return losses.mean(), cos_sim_pos.mean(), cos_sim_neg.mean()