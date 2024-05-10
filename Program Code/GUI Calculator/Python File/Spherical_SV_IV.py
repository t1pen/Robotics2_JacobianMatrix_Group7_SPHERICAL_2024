import numpy as np
import sympy as sp

# Link lengths in mm
a1 = float(input("a1 = "))
a2 = float(input("a2 = "))
a3 = float(input("a3 = "))

# Joint variables: in mm if d, in degrees if theta
T1 = float(input("T1 = "))  # 0
T2 = float(input("T2 = "))  # 90 deg
d3 = float(input("d3 = "))  # 3

# Convert Rotation angles (deg to rad)
T1 = np.radians(T1)
T2 = np.radians(T2)

# Parametric Table
PT = [[T1, np.pi/2, 0, a1],
      [T2 + np.pi/2, np.pi/2, 0, 0],
      [0, 0, 0, a2 + a3 + d3]]

# Homogeneous Transformation Matrix Formula
def compute_H_matrix(theta, alpha, a, d):
    return np.array([[np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
                     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
                     [0, np.sin(alpha), np.cos(alpha), d],
                     [0, 0, 0, 1]])

H0_1 = compute_H_matrix(PT[0][0], PT[0][1], PT[0][2], PT[0][3])
H1_2 = compute_H_matrix(PT[1][0], PT[1][1], PT[1][2], PT[1][3])
H2_3 = compute_H_matrix(PT[2][0], PT[2][1], PT[2][2], PT[2][3])

H0_1 = np.matrix(H0_1)
print("H0_1 = ")
print(H0_1)

H1_2 = np.matrix(H1_2)
print("H1_2 = ")
print(H1_2)

H2_3 = np.matrix(H2_3)
print("H2_3 = ")
print(H2_3)

H0_2 = np.dot(H0_1,H1_2)
H0_3 = np.dot(H0_2,H2_3)
print("H0_3 = ")
print(np.matrix(H0_3))
H0_3 = np.matrix(H0_3)

# Jacobian Matrix

# 1. Linear/Translation Vectors
Z_1 = np.array([[0], [0], [1]])  # The [0,0,1] vector

# Row 1 to 3, Column 1
J1a = np.dot(np.array([[1,0,0], [0,1,0], [0,0,1]]), Z_1)

J1b_1 = H0_3[0:3,3:] # d0_3
J1b_2 = H0_1[0:3,3:] # d0_1
J1b = J1b_1 - J1b_2

J1 = np.array([[J1a[1,0]*J1b[2,0] - J1a[2,0]*J1b[1,0]],
               [J1a[2,0]*J1b[0,0] - J1a[0,0]*J1b[2,0]],
               [J1a[0,0]*J1b[1,0] - J1a[1,0]*J1b[0,0]]])

J1 = np.matrix(J1)

# Row 1 to 3, Column 2
J2a = np.dot(H0_1[0:3,0:3], Z_1)

J2b_1 = H0_3[0:3,3:] # d0_3
J2b_2 = H0_1[0:3,3:] # d0_1
J2b = J2b_1 - J2b_2

J2 = np.array([[J2a[1,0]*J2b[2,0] - J2a[2,0]*J2b[1,0]],
               [J2a[2,0]*J2b[0,0] - J2a[0,0]*J2b[2,0]],
               [J2a[0,0]*J2b[1,0] - J2a[1,0]*J2b[0,0]]])

J2 = np.matrix(J2)

# Row 1 to 3, Column 3
J3a = np.dot(H0_2[0:3,0:3], Z_1)
J3 = np.matrix(J3a)

# 2. Rotation/Orientation Vectors
# Row 4 to 6, Column 1
J4 = J1a

# Row 4 to 6, Column 2
J5 = J2a

J6 = np.array([[0], [0], [0]])

# 3. Concatenated Jacobian Matrix
JM1 = np.concatenate((J1, J2, J3), axis=1)
JM2 = np.concatenate((J4, J5, J6), axis=1)
J = np.concatenate((JM1, JM2), axis=0)

print("Jacobian Matrix J = ")
print(np.around(J, 3))

# 4. Differential Equations
T1_p, T2_p, d3_p = sp.symbols('θ₁* θ₂* d₃*')

q = [[T1_p], [T2_p], [d3_p]]

E = np.dot(J, q)
E = np.matrix(E)

xp = E[0,0]
yp = E[1,0]
zp = E[2,0]
ωx = E[3,0]
ωy = E[4,0]
ωz = E[5,0]

print("xp = ", xp)
print("yp = ", yp)
print("zp = ", zp)
print("ωx = ", ωx)
print("ωy = ", ωy)
print("ωz = ", ωz)

## Singularity
D_J = np.linalg.det(JM1)
print("D_J = ", D_J)

xp, yp, zp, ωx, ωy, ωz = sp.symbols('xp yp zp ωx ωy ωz')

E = np.array([[xp, yp, zp, ωx, ωy, ωz]])

# Inverse Jacobian
J_inv = np.linalg.pinv(J)

# Define symbols for velocities
T1_p, T2_p, d3_p = sp.symbols('T1_p T2_p d3_p')

# Define the velocity vector
q_p = sp.Matrix([[T1_p], [T2_p], [d3_p], [0], [0], [0]])  # Include zeros for the angular velocities

# Calculate the joint velocities
q_dot = J_inv * q_p

# Substitute numerical values into joint velocities
q_dot_num = q_dot.subs({T1_p: T1, T2_p: T2, d3_p: d3})

# Convert joint velocities to numpy array for printing
joint_velocities = np.array(q_dot_num).astype(float)

T1_prime = joint_velocities[0]
T2_prime = joint_velocities[1]
d3_prime = joint_velocities[2]

print("Joint velocities (cm/s and rad/s):")
print("θ₁' = ", T1_prime, "rad/s")
print("θ₂' = ", T2_prime, "rad/s")
print("d₃' = ", d3_prime, "cm/s")
