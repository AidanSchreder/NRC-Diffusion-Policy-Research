import json
import matplotlib.pyplot as plt

log_file_path = '/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/Inference/Checkpoints/2026.02.13/14.13.38_unet_wiping/logs.json.txt'

train_steps = []
train_loss = []

val_epochs = []
val_loss = []

with open(log_file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)

        if "train_loss" in data and "global_step" in data:
            train_steps.append(data["global_step"])
            train_loss.append(data["train_loss"])
        if "val_loss" in data and "epoch" in data:
            val_epochs.append(data["epoch"])
            val_loss.append(data["val_loss"])
            
        
print(f"Total Training Steps: {len(train_steps)}")
print(f"Total Epochs: {len(val_epochs)}")
print(f"Minimum val_loss: {min(val_loss)}, Index of minimum val_loss: {val_loss.index(min(val_loss))}")
plt.figure()
plt.plot(train_steps, train_loss)
plt.xlabel("Step")
plt.ylabel("Train Loss")
plt.title("train_loss")


plt.figure()
plt.plot(val_epochs, val_loss)
plt.xlabel("Epoch")
plt.ylabel("Val Loss")
plt.title("val_loss")

plt.show()