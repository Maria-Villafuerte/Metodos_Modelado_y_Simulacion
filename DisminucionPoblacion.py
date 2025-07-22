# Modelo de decrecimiento poblacional
import numpy as np
import matplotlib.pyplot as plt

# Valores iniciales
poblacion = 100     # población inicial
tiempo_total = 24
tasa_mortalidad = 0.1 #por mes

# Método de Euler
t = 0.1             # Paso de tiempo
mes = 0
p_euler = [poblacion]           #historial de población

while mes < tiempo_total:
    p_anterior = p_euler[-1]
    dp_dt = -tasa_mortalidad*p_anterior
    p_euler.append(p_anterior + t*dp_dt) # calculo de poblacion actual
    mes += t #paso del tiempo

# Solución analítica
tiempos = np.arange(0, tiempo_total + t, t)# Tiempo
p_analitica = poblacion * np.exp(-tasa_mortalidad * tiempos) # calculo directo

# Graficar comparación
plt.plot(tiempos, p_euler, label="Método de Euler", linestyle='--', marker='o', color='skyblue')
plt.plot(tiempos, p_analitica, label="Solución Analítica", linewidth=2)
plt.xlabel("Tiempo (meses)")
plt.ylabel("Población")
plt.title("Decrecimiento poblacional: Euler vs. Solución exacta")
plt.legend()
plt.grid(True)
plt.show()