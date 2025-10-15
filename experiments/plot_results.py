# import torch
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load checkpoint
# checkpoint = torch.load("rnn_padded.pth", weights_only=False)

# # Extract results
# results = checkpoint['results']
# df = pd.DataFrame(results)

# # Plot Accuracy
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# sns.lineplot(x='epoch', y='train Accuracy', data=df, label='Train')
# sns.lineplot(x='epoch', y='test Accuracy', data=df, label='Test')
# plt.title("Accuracy per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.legend()

# # Plot Loss
# plt.subplot(1, 2, 2)
# sns.lineplot(x='epoch', y='train loss', data=df, label='Train')
# sns.lineplot(x='epoch', y='test loss', data=df, label='Test')
# plt.title("Loss per Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.show()
# # Save plots
# plt.savefig("training_results.png")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# Load checkpoint
checkpoint = torch.load("final_best_model.pth", weights_only=False)

# Extract results
results = checkpoint["results"]
df = pd.DataFrame(results)

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.lineplot(x="epoch", y="train Accuracy", data=df, label="Train")
sns.lineplot(x="epoch", y="test Accuracy", data=df, label="Test")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
sns.lineplot(x="epoch", y="train loss", data=df, label="Train")
sns.lineplot(x="epoch", y="test loss", data=df, label="Test")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
# Save plots
plt.savefig("training_results.png")
