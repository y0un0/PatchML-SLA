import torch
import torch.nn as nn
import torch.nn.functional as F

class WeakNegativeLoss(nn.Module):
    def __init__(self, device, theta=0.0):
        super(WeakNegativeLoss, self).__init__()
        # Theta is the parameter for the ReLU
        self.theta = theta
        self.device = device

    def threshold_relu(self, x):
        return torch.where(x > self.theta, x, torch.zeros_like(x))

    def forward(self, image_embeddings, predicted_probs, positive_labels):
        """
        @param image_embeddings: Tensor of shape (batch_size, num_labels, embedding_dim), the image representations for each label
        @param positive_labels: Tensor of shape (batch_size, num_labels), binary (1 for positive labels, 0 otherwise)
        @param predicted_probs: Tensor of shape (batch_size, num_labels), output probabilities (after applying softmax/sigmoid)

        @return loss: Scalar loss value
        """
        batch_size, num_labels, _ = image_embeddings.shape

        eps = 1e-7
        positive_loss = 0
        negative_loss = 0
        for i in range(batch_size):
            # Clamp predicted probabilities for numerical stability
            pred_probs = torch.clamp(predicted_probs[i], min=eps, max=1 - eps)
            positive_label_index = torch.argwhere(positive_labels[i]).item()
            # Positive loss: -z⁺ log(ŷ)
            pos_log = torch.log(pred_probs)
            positive_loss += -torch.sum(positive_labels[i] * pos_log)
            if torch.isnan(pos_log).any():
                print(f"NaN in positive log probabilities for sample {i}: {pos_log}")

            normalized_embeddings = F.normalize(image_embeddings[i], p=2, dim=1)
            # Cosine similarity matrix between all pairs of label embeddings to find the negative labels
            cos_sim_matrix = normalized_embeddings @ normalized_embeddings.T
            if torch.isnan(cos_sim_matrix).any():
                print(f"NaN in cosine similarity matrix for sample {i}: {cos_sim_matrix}")
            
            # Weak negative labels estimation
            beta_matrix = self.threshold_relu(cos_sim_matrix)
            weak_negative_labels = torch.zeros(num_labels, device=self.device)
            for l in range(num_labels):
                if l == positive_label_index:
                    # For the positive label, ensure the negative score is zero
                    weak_negative_labels[l] = 0.0
                else:
                    beta_lk = beta_matrix[l].clone()
                    beta_value = beta_lk[positive_label_index]
                    weak_negative_labels[l] = beta_value
            # Negative loss: - z⁻ log(1 - ŷ)
            neg_log = torch.log(1 - pred_probs)
            negative_loss += -torch.sum(weak_negative_labels * neg_log)
            if torch.isnan(neg_log).any():
                print(f"NaN in negative log probabilities for sample {i}: {neg_log}")

        # Weak negative loss
        weak_negative_loss = (positive_loss + negative_loss) / batch_size
        if torch.isnan(weak_negative_loss):
            print(f"NaN in weak negative loss: {weak_negative_loss}")
        return weak_negative_loss, positive_loss / batch_size, negative_loss / batch_size
    
if __name__ == "__main__":
    batch_size = 1
    num_labels = 4
    embedding_dim = 256

    # Simulated image embeddings and label embeddings
    image_embeddings = torch.randn(batch_size, num_labels, embedding_dim)  # (batch_size, num_labels, embedding_dim)

    # Simulated positive labels (z⁺) (one-hot encoded or binary labels per class)
    # positive_labels = torch.randint(0, 2, (batch_size, num_labels)).float()  # Shape: (batch_size, num_labels)
    positive_labels = torch.Tensor([0.0, 0.0, 1.0, 0.0]).float().unsqueeze(0)

    # Simulated predicted probabilities (after softmax or sigmoid)
    # predicted_probs = torch.sigmoid(torch.randn(batch_size, num_labels))  # Shape: (batch_size, num_labels)
    predicted_probs = torch.Tensor([0.9, 0.9, 0.7, 0.9]).float().unsqueeze(0)

    # Instantiate the loss function and compute the loss
    loss_fn = WeakNegativeLoss("cpu", theta=0.0)
    loss, positive_loss, negative_loss = loss_fn(image_embeddings, predicted_probs, positive_labels)

    print(f"Loss: {loss.item()}")
    print(f"Positive Loss: {positive_loss.item()}")
    print (f"Negative Loss: {negative_loss.item()}")