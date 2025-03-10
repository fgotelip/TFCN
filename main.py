import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate

def calcular_g(N, a, b, m=1000):
    """
    Calcula os valores g_j numericamente usando Newton-Cotes repetido.
    """
    x = np.linspace(a, b, m+1)
    h = (b - a) / m
    g = np.array([np.trapezoid(x**(j-1), x) for j in range(1, 2*N+1)])
    return g

def f(w, t, g, N):
    """
    Função
    """
    return np.array([np.sum(w * t**(j-1)) - g[j-1] for j in range(1, 2*N+1)])

def jacobiana(w, t, g, N, epsilon=1e-8):
    """
    Calcula a matriz Jacobiana numericamente.
    """
    J = np.zeros((2*N, 2*N))
    f0 = f(w, t, g, N)
    for i in range(N):
        w_eps = w.copy()
        w_eps[i] += epsilon
        J[:, i] = (f(w_eps, t, g, N) - f0) / epsilon
        
        t_eps = t.copy()
        t_eps[i] += epsilon
        J[:, N + i] = (f(w, t_eps, g, N) - f0) / epsilon
    
    return J

def metodo_newton(N, a, b, tol=1e-8, max_iter=100):
    """
    Resolve o sistema não linear pelo método de Newton para obter os pontos e pesos da quadratura de Gauss.
    """
    # Condições iniciais sugeridas
    w = np.array([(b - a) / (2 * N) if i < N/2 else (b - a) / (2 * N) for i in range(N)])
    t = np.array([a + i * w[i] / 2 if i < N/2 else (a + b) - (a + (i - N//2) * w[i] / 2) for i in range(N)])
    if N % 2 == 1:
        t[N//2] = (a + b) / 2
    
    g = calcular_g(N, a, b)
    
    for _ in range(max_iter):
        f_val = f(w, t, g, N)
        if np.linalg.norm(f_val, np.inf) < tol:
            break
        
        J = jacobiana(w, t, g, N)
        s = la.solve(J, -f_val)
        
        w += s[:N]
        t += s[N:]
    
    return t, w

def integrar_gauss(f, N, a, b):
    """
    Usa quadratura de Gauss para integrar numericamente a função f no intervalo [a, b].
    """
    t, w = metodo_newton(N, a, b)
    return np.sum(w * f(t))

def calcular_erro(f, N, a, b):
    """
    Calcula a integral exata, a integral aproximada e o erro absoluto.
    """
    integral_exata, _ = integrate.quad(f, a, b)
    integral_aproximada = integrar_gauss(f, N, a, b)
    erro = abs(integral_exata - integral_aproximada)
    
    return integral_exata, integral_aproximada, erro


a, b = -1, 1  # Intervalo de integração
N = 3  # Número de pontos de integração

pontos, pesos = metodo_newton(N, a, b)
print("Pontos de integração:", pontos)
print("Pesos:", pesos)

# Testando a integração numérica
funcao = lambda x: np.exp(-x**2)  # Exemplo de função

# Calculando erro
integral_exata, integral_aproximada, erro = calcular_erro(funcao, N, a, b)
print(f"Integral exata: {integral_exata}")
print(f"Integral aproximada: {integral_aproximada}")
print(f"Erro absoluto: {erro}")
