import numpy as np
import matplotlib.pyplot as plt
import time

# Parámetros
r = 0.1
K = 1000
x0 = 10

# Definimos Ecuación logística
def ecuacion_logica(x):
    return r * x * (1 - x/K)

# Solución analítica
def exacta(t):
    return K / (1 + ((K-x0)/x0) * np.exp(-r*t))

# Método de Euler
def metodo_euler(dt, t_max):
    t = np.arange(0, t_max+dt, dt)
    x = np.zeros(len(t))
    x[0] = x0
    
    inicio = time.time()
    for i in range(len(t)-1):
        x[i+1] = x[i] + dt * ecuacion_logica(x[i])
    tiempo = time.time() - inicio
    
    return t, x, tiempo

# Método RK4
def metodo_rk4(dt, t_max):
    t = np.arange(0, t_max+dt, dt)
    x = np.zeros(len(t))
    x[0] = x0
    
    inicio = time.time()
    for i in range(len(t)-1):
        k1 = ecuacion_logica(x[i])
        k2 = ecuacion_logica(x[i] + dt*k1/2)
        k3 = ecuacion_logica(x[i] + dt*k2/2)
        k4 = ecuacion_logica(x[i] + dt*k3)
        x[i+1] = x[i] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    tiempo = time.time() - inicio
    
    return t, x, tiempo

print("COMPARACIÓN EULER vs RK4")
print("="*40)

# a) Error relativo en t=50 con dt=1.0
print("\na) Error en t=50 (Δt=1.0):")
t_e, x_e, _ = metodo_euler(1.0, 50)
t_r, x_r, _ = metodo_rk4(1.0, 50)
x_exacta = exacta(50)

error_euler = abs(x_e[-1] - x_exacta) / x_exacta * 100
error_rk4 = abs(x_r[-1] - x_exacta) / x_exacta * 100

print(f"Solución exacta: {x_exacta:.2f}")
print(f"Euler: {x_e[-1]:.2f} (error: {error_euler:.4f}%)")
print(f"RK4: {x_r[-1]:.2f} (error: {error_rk4:.4f}%)")

# b) Tiempo vs precisión
print("\nb) Tiempo vs Precisión:")
print("Δt\tError Euler\tTiempo E\tError RK4\tTiempo RK4")
for dt in [0.1, 0.5, 1.0, 2.0, 5.0]:
    t_e, x_e, tiempo_e = metodo_euler(dt, 100)
    t_r, x_r, tiempo_r = metodo_rk4(dt, 100)
    x_final = exacta(100)
    
    err_e = abs(x_e[-1] - x_final) / x_final * 100
    err_r = abs(x_r[-1] - x_final) / x_final * 100
    
    print(f"{dt}\t{err_e:.4f}%\t{tiempo_e:.6f}s\t{err_r:.4f}%\t{tiempo_r:.6f}s")

# c) Estabilidad con dt grandes
print("\nc) Estabilidad con Δt > 5.0:")
for dt in [5.0, 10.0, 15.0, 20.0]:
    t_e, x_e, _ = metodo_euler(dt, 100)
    t_r, x_r, _ = metodo_rk4(dt, 100)
    
    # Verificar estabilidad (valores negativos o muy grandes)
    estable_e = "Estable" if (x_e.min() >= 0 and x_e.max() < 2*K) else "INESTABLE"
    estable_r = "Estable" if (x_r.min() >= 0 and x_r.max() < 2*K) else "INESTABLE"
    
    print(f"Δt={dt}: Euler={estable_e}, RK4={estable_r}")

# Gráfico 
plt.figure(figsize=(10, 6))

# Calcular soluciones para graficar
t_e, x_e, _ = metodo_euler(1.0, 100)
t_r, x_r, _ = metodo_rk4(1.0, 100)
t_exact = np.linspace(0, 100, 500)
x_exact = exacta(t_exact)

plt.plot(t_exact, x_exact, 'k-', label='Exacta', linewidth=2)
plt.plot(t_e, x_e, 'r--', label='Euler', alpha=0.7)
plt.plot(t_r, x_r, 'b:', label='RK4', alpha=0.7)
plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.title('Crecimiento Logístico: Comparación de Métodos')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()