import torch
import torch.nn as nn
import torch.nn.functional as F

class WeakNegativeLoss(nn.Module):
    def __init__(self, theta=0.0):
        super(WeakNegativeLoss, self).__init__()
        # Theta is the parameter for the ReLU
        self.theta = theta

    def threshold_relu(self, x):
        return F.relu(x - self.theta)

    def forward(self, image_embeddings, predicted_probs, positive_labels):
        """
        @param image_embeddings: Tensor of shape (batch_size, num_labels, embedding_dim), the image representations for each label
        @param positive_labels: Tensor of shape (batch_size, num_labels), binary (1 for positive labels, 0 otherwise)
        @param predicted_probs: Tensor of shape (batch_size, num_labels), output probabilities (after applying softmax/sigmoid)

        @return loss: Scalar loss value
        """
        batch_size, num_labels, _ = image_embeddings.shape
        
        positive_loss = 0
        negative_loss = 0
        for i in range(batch_size):
            # Positive loss: -z⁺ log(ŷ) (CrossEntropy)
            positive_loss += -torch.sum(positive_labels[i] * torch.log(predicted_probs[i] + 1e-7))
            # Cosine similarity matrix between all pairs of label embeddings to find the negative labels
            cos_sim_matrix = F.cosine_similarity(image_embeddings[i].unsqueeze(1), image_embeddings[i].unsqueeze(0), dim=-1)
            
            # Weak negative labels estimation
            beta_matrix = self.threshold_relu(cos_sim_matrix)
            weak_negative_labels = torch.zeros(num_labels)
            for l in range(num_labels):
                # We need to go through all unobserved labels (z⁺_l = 0)
                if positive_labels[i, l] == 0:
                    # Find the max beta_{l,k} for all k where z⁺_k = 1
                    beta_lk = beta_matrix[l] * positive_labels[i]  # Mask to only consider positive labels
                    weak_negative_labels[l] = torch.max(beta_lk)
            
            # Negative loss: - z⁻ log(1 - ŷ)
            negative_loss += -torch.sum(weak_negative_labels[i] * torch.log(1 - (predicted_probs[i] + 1e-7)))
        
        # Weak negative loss
        weak_negative_loss = (positive_loss + negative_loss) / batch_size
        return weak_negative_loss
    
if __name__ == "__main__":
    batch_size = 1
    num_labels = 5
    embedding_dim = 256

    # Simulated image embeddings and label embeddings
    image_embeddings = torch.randn(batch_size, num_labels, embedding_dim)  # (batch_size, num_labels, embedding_dim)

    # Simulated positive labels (z⁺) (one-hot encoded or binary labels per class)
    positive_labels = torch.randint(0, 2, (batch_size, num_labels)).float()  # Shape: (batch_size, num_labels)
    # Simulated predicted probabilities (after softmax or sigmoid)
    predicted_probs = torch.sigmoid(torch.randn(batch_size, num_labels))  # Shape: (batch_size, num_labels)

    # Instantiate the loss function and compute the loss
    loss_fn = WeakNegativeLoss(theta=0.0)
    loss = loss_fn(image_embeddings, predicted_probs, positive_labels)

    print(f"Loss: {loss.item()}")