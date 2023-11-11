# Digital-Signal-Processing-DSP-

# Z transform-------
```ruby
import control
import matplotlib.pyplot as plt

# Define a continuous-time transfer function
f = control.tf([1, 1], [1, 1, 1])
print("Continuous-time transfer function:", f)

# Convert the continuous-time transfer function to discrete-time with a sampling time of 1 second
z = control.c2d(f, 1)
print("Discrete-time transfer function:", z)

t, h = control.impulse_response(f)
plt.stem(t, h)
plt.show()

# Plot poles and zeros of the discrete-time transfer function
control.pzmap(z, True, 'Poles and Zeros')
plt.title('Poles and Zeros')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)
plt.show()

# Calculate and print the poles of the discrete-time transfer function
poles = control.pole(z)
print("Poles:", poles)
zero = control.zeros(z)
print("Zeroes:", zero)
```
