import torch
import numpy as np

tensor = torch.rand(2,3)
matrix = np.random.rand(2,3)
print(f"Tensor:\n {tensor}")
print(f"Matrix:\n {matrix}")

tensor = torch.zeros(2,3)
matrix = np.zeros((2,3))
print(f"Tensor:\n {tensor}")
print(f"Matrix:\n {matrix}")

tensor = torch.ones(2,3)
matrix = np.ones((2,3))
print(f"Tensor:\n {tensor}")
print(f"Matrix:\n {matrix}")

print(f"Tensor:\n {tensor.dtype}")
print(f"Matrix:\n {matrix.dtype}")

print(f"Tensor:\n {tensor.size()}")
print(f"Tensor:\n {tensor.shape}")
print(f"Matrix:\n {matrix.shape}")

# list to torch
tensor = torch.tensor([2.3, 1.2, 5.7])
print(f"Tensor:\n {tensor.size()}")

# list to numpy
matrix = np.array([2.3, 1.2, 5.7])
print(f"Matrix:\n {matrix.shape}")

# numpy <=> torch
tensor = torch.tensor([1, 2, 3])
matrix = np.array([1, 2, 3])
print(torch.from_numpy(matrix))
print(tensor.numpy())

# call by reference
tensor = torch.tensor([1, 2, 3])
matrix = tensor.numpy()

tensor += 1
print(tensor)
print(matrix)

# operator
t1 = torch.tensor([2.3, 1.2, 5.7])
t2 = torch.tensor([3.2, 2.8, 7.8])
print(torch.add(t1, t2)) # t1 + t2
print(torch.add(t1, t2)) # t1 - t2
print(torch.mul(t1, t2)) # t1 * t2
print(torch.div(t1, t2)) # t1 / t2

# slice
tensor = torch.rand(5,3)
print(tensor)
print(tensor[:,1:3])
print(tensor[1,1].item())

# view
tensor = torch.rand(2,6)
print(tensor)
print(tensor.view(3,4))
print(tensor.view(12))
print(tensor.view(12,1))
print(tensor.view(1,12))
print(tensor.view(-1,3))

# GPU
print(torch.cuda.is_available())
# if torch.cuda.is_available():
#   device = torch.device("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.rand(3,2).to(device)
print(x)
x = x.to("cpu")
print(x)