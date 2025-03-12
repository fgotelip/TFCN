import numpy as np
from scipy.integrate import quad

# Função para calcular as integrais g_j


def calcular_gj(a, b, j, m=1000):
    def integrando(x):
        return x ** (j - 1)

    resultado, _ = quad(integrando, a, b, limit=m)
    return resultado

# Função para calcular as condições iniciais w0 e t0


def condicoes_iniciais(a, b, N):
    w0 = np.zeros(N)
    t0 = np.zeros(N)

    for i in range(N):
        if i < N / 2:
            w0[i] = (b - a) / (2 * N) * (i + 1)
            t0[i] = a + (i + 1) * w0[i] / 2
        else:
            w0[i] = w0[int(N / 2) - (i - int(N / 2)) - 1]
            t0[i] = (a + b) - t0[int(N / 2) - (i - int(N / 2)) - 1]

    if N % 2 != 0:
        t0[int(N / 2)] = (a + b) / 2

    return w0, t0

# Função para calcular o vetor f(w, t)


def calcular_f(w, t, g):
    N = len(w)
    f = np.zeros(2 * N)

    for j in range(1, 2 * N + 1):
        soma = 0
        for i in range(N):
            soma += w[i] * (t[i] ** (j - 1))
        f[j - 1] = soma - g[j - 1]

    return f

# Função para calcular a matriz Jacobiana


def calcular_jacobiana(w, t, epsilon=1e-8):
    N = len(w)
    J = np.zeros((2 * N, 2 * N))

    for j in range(2 * N):
        for i in range(N):
            # Derivada em relação a w[i]
            w_perturbado = w.copy()
            w_perturbado[i] += epsilon
            f_perturbado_w = calcular_f(w_perturbado, t, g)
            f_original = calcular_f(w, t, g)
            J[j, i] = (f_perturbado_w[j] - f_original[j]) / epsilon

            # Derivada em relação a t[i]
            t_perturbado = t.copy()
            t_perturbado[i] += epsilon
            f_perturbado_t = calcular_f(w, t_perturbado, g)
            J[j, N + i] = (f_perturbado_t[j] - f_original[j]) / epsilon

    return J

# Método de Newton para resolver o sistema não linear


def metodo_newton(a, b, N, TOL=1e-8, max_iter=100):
    w0, t0 = condicoes_iniciais(a, b, N)
    w = w0.copy()
    t = t0.copy()

    # Calcular o vetor g
    g = np.zeros(2 * N)
    for j in range(1, 2 * N + 1):
        g[j - 1] = calcular_gj(a, b, j)

    iteracao = 0
    while iteracao < max_iter:
        f = calcular_f(w, t, g)
        if np.linalg.norm(f, np.inf) < TOL:
            break

        J = calcular_jacobiana(w, t)
        s = np.linalg.solve(J, -f)

        w += s[:N]
        t += s[N:]

        iteracao += 1

    return w, t


# Exemplo de uso
a = -1
b = 1
N = 4
w, t = metodo_newton(a, b, N)
print("Pesos (w):", w)
print("Pontos de integração (t):", t)
