import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import onnx

class _TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = _TestNet()
model.eval()

# Save as TorchScript
scripted = torch.jit.trace(model, torch.randn(1, 10))
scripted.save("test_model_scripted.pt")
print("✅ test_model_scripted.pt created")

# Save as ONNX (self-contained)
torch.onnx.export(
    model, torch.randn(1, 10), "test_model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
onnx_model = onnx.load("test_model.onnx")
onnx.save_model(onnx_model, "test_model.onnx", save_as_external_data=False)
print("✅ test_model.onnx created")

# Dataset
X = np.random.rand(200, 10)
y = np.random.randint(0, 2, 200)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
df["label"] = y
df.to_csv("test_dataset.csv", index=False)
print("✅ test_dataset.csv created")