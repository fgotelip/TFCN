import numpy as np
from scipy.linalg import solve

def newton_cotes_integral(a, b, j, m=1000):
    """
    Calcula numericamente a integral de x^(j-1) no intervalo [a, b]
    usando a regra de Newton-Cotes composta.
    """
    x = np.linspace(a, b, m)
    dx = (b - a) / (m - 1)
    integral = np.sum(x**(j-1)) * dx
    return integral

def inicializar_w_t(N, a, b):
    """
    Inicializa os pesos w0 e pontos t0 conforme especificado.
    """
    w0 = np.zeros(N)
    t0 = np.zeros(N)
    
    for i in range(N):
        if i < N / 2:
            w0[i] = ((b - a) / (2 * N)) * (i + 1)
        else:
            w0[i] = w0[int(N/2) - (i - int(N/2)) - 1]
    
    for i in range(N):
        if i < N / 2:
            t0[i] = a + i * w0[i] / 2
        else:
            t0[i] = (a + b) - t0[int(N/2) - (i - int(N/2)) - 1]
    
    if N % 2 == 1:
        t0[N//2] = (a + b) / 2
    
    return w0, t0

def calcular_jacobiana(N, w, t, a, b, epsilon=1e-8):
    """
    Calcula a matriz Jacobiana numericamente.
    """
    J = np.zeros((2*N, 2*N))
    f_val = calcular_f(N, w, t, a, b)
    
    for i in range(2*N):
        for j in range(2*N):
            pert = np.zeros(2*N)
            pert[j] = epsilon
            
            if j < N:
                w_pert = w + pert[:N]
                t_pert = t
            else:
                w_pert = w
                t_pert = t + pert[N:]
            
            f_pert = calcular_f(N, w_pert, t_pert, a, b)
            J[i, j] = (f_pert[i] - f_val[i]) / epsilon
    
    return J

def calcular_f(N, w, t, a, b):
    """
    Calcula o vetor f para o sistema não linear.
    """
    f = np.zeros(2*N)
    
    for j in range(1, 2*N+1):
        soma = np.sum(w * (t ** (j-1)))
        g = newton_cotes_integral(a, b, j)
        f[j-1] = soma - g
    
    return f

def metodo_newton(N, a, b, tol=1e-8, max_iter=100):
    """
    Método de Newton para encontrar os pontos e pesos da quadratura de Gauss.
    """
    w, t = inicializar_w_t(N, a, b)
    
    for _ in range(max_iter):
        f = calcular_f(N, w, t, a, b)
        if np.linalg.norm(f, np.inf) < tol:
            break
        
        J = calcular_jacobiana(N, w, t, a, b)
        s = solve(J, -f)
        
        w += s[:N]
        t += s[N:]
    
    return w, t

# Exemplo de uso
a, b = -1, 1
N = 4
pesos, pontos = metodo_newton(N, a, b)
print("Pesos:", pesos)
print("Pontos:", pontos)
