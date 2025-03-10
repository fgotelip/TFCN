import numpy as np

def pontos_e_pesos_gauss_legendre(a, b, N, tol=1e-8, epsilon=1e-8, m=1000):
    # Função para calcular as integrais g_j
    def calcular_g(j):
        return (b**j - a**j) / j

    # Função para calcular f_j(w, t)
    def calcular_f(w, t, j):
        return np.sum(w * t**(j-1)) - calcular_g(j)

    # Função para calcular a matriz Jacobiana
    def calcular_jacobiano(w, t):
        J = np.zeros((2*N, 2*N))
        for j in range(1, 2*N+1):
            for i in range(N):
                # Derivada em relação a w_i
                w_perturbado = w.copy()
                w_perturbado[i] += epsilon
                f_perturbado = calcular_f(w_perturbado, t, j)
                f_original = calcular_f(w, t, j)
                J[j-1, i] = (f_perturbado - f_original) / epsilon

                # Derivada em relação a t_i
                t_perturbado = t.copy()
                t_perturbado[i] += epsilon
                f_perturbado = calcular_f(w, t_perturbado, j)
                f_original = calcular_f(w, t, j)
                J[j-1, N+i] = (f_perturbado - f_original) / epsilon
        return J

    # Condições iniciais para w e t
    w0 = np.zeros(N)
    t0 = np.zeros(N)
    for i in range(N):
        if i < N/2:
            w0[i] = (b - a) / (2 * N) * (i + 1)
            t0[i] = a + i * w0[i] / 2
        else:
            w0[i] = w0[int(N/2) - (i - int(N/2)) - 1]
            t0[i] = (a + b) - t0[int(N/2) - (i - int(N/2)) - 1]
    if N % 2 != 0:
        t0[int(N/2)] = (a + b) / 2

    w = w0.copy()
    t = t0.copy()

    # Método de Newton
    while True:
        f = np.array([calcular_f(w, t, j) for j in range(1, 2*N+1)])
        J = calcular_jacobiano(w, t)
        delta = np.linalg.solve(J, -f)
        w_novo = w + delta[:N]
        t_novo = t + delta[N:]
        if np.linalg.norm(delta, np.inf) < tol:
            break
        w = w_novo
        t = t_novo

    return w, t

# Função para calcular a integral exata de uma função
def integral_exata(f, a, b):
    from scipy.integrate import quad
    resultado, _ = quad(f, a, b)
    return resultado

# Função para calcular a integral aproximada usando os pontos e pesos de Gauss-Legendre
def integral_aproximada(f, w, t):
    return np.sum(w * f(t))

# Função para calcular o erro entre a integral exata e a aproximada
def calcular_erro(integral_exata, integral_aproximada):
    return abs(integral_exata - integral_aproximada)

# Exemplo de uso
a = 0
b = 3
N = 6
w, t = pontos_e_pesos_gauss_legendre(a, b, N)
print("Pontos de integração (t):", t)
print("Pesos (w):", w)

# Definindo uma função de exemplo: f(x) = exp(x)
def f(x):
    return np.exp(x)

# Calculando a integral exata
integral_exata_valor = integral_exata(f, a, b)
print("Integral exata:", integral_exata_valor)

# Calculando a integral aproximada
integral_aproximada_valor = integral_aproximada(f, w, t)
print("Integral aproximada:", integral_aproximada_valor)

# Calculando o erro
erro = calcular_erro(integral_exata_valor, integral_aproximada_valor)
print("Erro:", erro)