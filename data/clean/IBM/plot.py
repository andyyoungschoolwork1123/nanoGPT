import matplotlib.pyplot as plt
iterations = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
]

train_losses = [
    10.8667, 1.1666, 0.6096, 0.5090, 0.4693, 0.4469, 0.4168, 0.3616, 0.3663, 0.3725, 0.3386
]

# Plotting the training loss vs iteration for the new data
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_losses, label='Training Loss (New)', color='red')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Iteration (New Data)')
plt.legend()
plt.grid(True)
plt.show()
# Plotting the training loss vs iteration
plt.figure(figsize=(10, 6))
plt.plot(iterations, train_losses, label='Training Loss', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show()