
import numpy as np

# Restricciones como matrices
# 3x1 + 4x2 <= 180
# 2x1 + 5x2 <= 160
A = np.array([[3, 4],
              [2, 5]])

b = np.array([180, 160])

# Coeficientes de la función objetivo
c = np.array([50, 70])

# Lista para guardar puntos factibles
puntos = []

# Intersecciones con los ejes
puntos.append([0, b[0]/A[0,1]])   # (0, 180/4)
puntos.append([b[0]/A[0,0], 0])   # (180/3, 0)
puntos.append([0, b[1]/A[1,1]])   # (0, 160/5)
puntos.append([b[1]/A[1,0], 0])   # (160/2, 0)

# Intersección entre las dos restricciones (resolver sistema lineal)
punto_inter = np.linalg.solve(A, b)
puntos.append(punto_inter)

# Filtrar puntos factibles (que cumplen restricciones y x>=0)
factibles = []
for p in puntos:
    x1, x2 = p
    if x1 >= 0 and x2 >= 0 and np.all(A @ [x1, x2] <= b + 1e-6):
        factibles.append([x1, x2])

factibles = np.array(factibles)

# Evaluar función objetivo
valores = factibles @ c

# Encontrar el mejor
idx = np.argmax(valores)
optimo = factibles[idx]
z_opt = valores[idx]

print("Puntos factibles:")
print(factibles)
print("\nMejor solución:")
print("x1 (Armarios) =", optimo[0])
print("x2 (Camas) =", optimo[1])
print("Z máximo =", z_opt)

# Commented out IPython magic to ensure Python compatibility.
# %pip install pulp

"""# Nueva sección"""

import numpy as np
import matplotlib.pyplot as plt

# Restricciones
# 3x1 + 4x2 <= 180
# 2x1 + 5x2 <= 160

x1 = np.linspace(0, 80, 400)

# Restricción 1 -> despejar x2 = (180 - 3x1)/4
x2_1 = (180 - 3*x1) / 4
# Restricción 2 -> despejar x2 = (160 - 2x1)/5
x2_2 = (160 - 2*x1) / 5

# Región factible
x2_factible = np.minimum(x2_1, x2_2)
x2_factible = np.maximum(x2_factible, 0)  # no negativos

# Definir función objetivo Z = 50x1 + 70x2
def Z(x1, x2):
    return 50*x1 + 70*x2

# Puntos de intersección (resolver sistema)
A = np.array([[3, 4], [2, 5]])
b = np.array([180, 160])
p_inter = np.linalg.solve(A, b)

# Puntos factibles candidatos
puntos = [
    [0, 0],
    [0, 180/4],   # intersección con eje y (restr 1)
    [180/3, 0],   # intersección con eje x (restr 1)
    [0, 160/5],   # intersección con eje y (restr 2)
    [160/2, 0],   # intersección con eje x (restr 2)
    p_inter       # intersección entre restr 1 y 2
]

factibles = []
for (x,y) in puntos:
    if x >= 0 and y >= 0 and (3*x+4*y <= 180) and (2*x+5*y <= 160):
        factibles.append([x,y])

factibles = np.array(factibles)
valores = factibles @ np.array([50,70])
idx = np.argmax(valores)
optimo = factibles[idx]
z_opt = valores[idx]

# GRAFICAR
plt.figure(figsize=(8,6))

# Líneas de restricción
plt.plot(x1, x2_1, label="3x1 + 4x2 ≤ 180")
plt.plot(x1, x2_2, label="2x1 + 5x2 ≤ 160")

# Región factible (relleno)
plt.fill_between(x1, 0, np.minimum(x2_1, x2_2), alpha=0.3, color="lightblue")

# Puntos factibles
plt.scatter(factibles[:,0], factibles[:,1], color="red", label="Puntos factibles")

# Punto óptimo
plt.scatter(optimo[0], optimo[1], color="green", s=100, marker="*", label=f"Óptimo Z={z_opt:.0f}")

# Estilo
plt.xlim(0,80)
plt.ylim(0,60)
plt.xlabel("x1 (Armarios)")
plt.ylabel("x2 (Camas)")
plt.title("Región factible y solución óptima")
plt.legend()
plt.grid(True)
plt.show()
