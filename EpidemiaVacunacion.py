import numpy as np
import matplotlib.pyplot as plt

# Parámetros
beta = 0.3
gamma = 0.1

# Requisitos
dt = 0.5
T = 180
N = int(T / dt)
t = np.linspace(0, T, N+1)

# Condiciones iniciales
S0 = 0.99
I0 = 0.01
R0 = 0.0

v_values = np.linspace(0, 0.1, 6)  

# Gráficas por cada v
for v in v_values:
    S = np.zeros(N+1)
    I = np.zeros(N+1)
    R = np.zeros(N+1)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for n in range(N):
        dS = -beta * S[n] * I[n] - v * S[n]
        dI = beta * S[n] * I[n] - gamma * I[n]
        dR = gamma * I[n] + v * S[n]
        
        S[n+1] = S[n] + dt * dS
        I[n+1] = I[n] + dt * dI
        R[n+1] = R[n] + dt * dR
    
    I_max = I.max()
    t_max = t[np.argmax(I)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, S, label='Susceptibles (S)')
    plt.plot(t, I, label='Infectados (I)')
    plt.plot(t, R, label='Recuperados (R)')
    plt.title(f"v = {v:.2f} – Máximo infectados: {I_max:.4f} en día {t_max:.1f}")
    plt.xlabel("Días")
    plt.ylabel("Proporción")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
