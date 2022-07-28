import torch
import torch.nn as nn
# grad = True
"""
tensor = torch.rand(3, requires_grad=True)
print(tensor)
y = tensor + 6
print(y)
z = 3*y**2 + 2
print(z)
z = z.mean()
print(z)
z.backward() # dz/d(tensor)
print(tensor.grad) # 2*tensor + 12
"""

"""
# grad = False
tensor = torch.rand(3, requires_grad=False)
print(tensor)

y = tensor + 6
print(y)
z = 3*y**2 + 2
print(z)
z = z.mean()
print(z)
z.backward() # dz/d(tensor)
"""

"""
# not scalar
tensor = torch.tensor([0.37, 0.58, 0.33], requires_grad=True)
print(tensor)

y = tensor + 6
print(y)
z = 3*y**2 + 2
print(z)
z.backward()
"""

"""
# grad 
tensor = torch.rand(3, requires_grad = True)
print(tensor)
y = tensor.detach()
print(y)
"""

"""
# grad 
with torch.no_grad():
  y = tensor + 2
  print(y)
"""

"""
# accumulated gradient
weights = torch.tensor([2., 3., 5., 7.], requires_grad=True)

for epoch in range(5):
  outputs = (3*weights).sum()
  outputs.backward()

  print(weights.grad)
"""

"""
# empty gradient
weights = torch.tensor([2., 3., 5., 7.], requires_grad=True)

for epoch in range(5):
  outputs = (3*weights).sum()
  outputs.backward()

  print(weights.grad)

  weights.grad.zero_()
"""

# toy example
x = torch.tensor([[1,-1], [2,3], [5,2]], dtype=torch.float32) # 3x2
y = torch.tensor([[1],[0],[1]], dtype=torch.float32)

w1 = torch.rand(2,3, requires_grad=True)
w2 = torch.rand(3,3, requires_grad=True)
w3 = torch.rand(3,2, requires_grad=True)
w4 = torch.rand(2,1, requires_grad=True)
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
bce = nn.BCELoss()

def forward(inputs):
    inputs = torch.matmul(inputs, w1)
    inputs = relu(inputs)
    inputs = torch.matmul(inputs, w2)
    inputs = relu(inputs)
    inputs = torch.matmul(inputs, w3)
    inputs = relu(inputs)
    inputs = torch.matmul(inputs, w4)
    outputs = sigmoid(inputs)
    return outputs

# loss
def loss(y_true, y_pred):
    return bce(y_pred, y_true)


learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # forward pass
    y_hat = forward(inputs=x)

    # loss
    bce_loss = loss(y_true=y, y_pred=y_hat)

    # backward loss
    bce_loss.backward()

    # update weights
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w3 -= learning_rate * w3.grad
        w4 -= learning_rate * w4.grad

    # zero gradients
    w1.grad.zero_()
    w2.grad.zero_()
    w3.grad.zero_()
    w4.grad.zero_()

    if epoch % 5 == 0:
        print(f"epoch {epoch + 1}: \nw1 = {w1}\n w2 = {w2}\n w3 = {w3}\n w4 = {w4}, loss = {bce_loss:.8f}")