import torch
import torchvision.models as models

# Load the pretrained EfficientNet-B0 model
model = models.efficientnet_b0(pretrained=True)
model.eval()  # set to evaluation mode

# Save the full model locally (this saves both architecture and weights)
torch.save(model, "efficientnet_b0_full.pt")

