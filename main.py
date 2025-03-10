import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate

def calcular_g(N, a, b, m=1000):
    """
    Calcula os valores g_j numericamente usando o método de Newton-Cotes repetido.
    """
    # Cria um vetor de pontos igualmente espaçados no intervalo [a, b]
    x = np.linspace(a, b, m+1)
    
    # Calcula o passo h
    h = (b - a) / m
    
    # Calcula os valores de g_j usando a regra do trapézio para cada j
    g = np.array([np.trapezoid(x**(j-1), x) for j in range(1, 2*N+1)])
    return g

def f(w, t, g, N):
    """
    Função que calcula a diferença entre a soma dos termos de quadratura e os valores g_j.
    """
    return np.array([np.sum(w * t**(j-1)) - g[j-1] for j in range(1, 2*N+1)])

def jacobiana(w, t, g, N, epsilon=1e-8):
    """
    Calcula a matriz Jacobiana numericamente.
    A Jacobiana é necessária para o método de Newton, que ajusta os valores de w e t.
    """
    J = np.zeros((2*N, 2*N))  # Inicializa a matriz Jacobiana
    f0 = f(w, t, g, N)  # Calcula os valores iniciais da função f(w, t)
    
    # Calcula a Jacobiana por diferenças finitas
    for i in range(N):
        # Perturba w[i] por uma pequena quantidade epsilon e calcula a diferença
        w_eps = w.copy()
        w_eps[i] += epsilon
        J[:, i] = (f(w_eps, t, g, N) - f0) / epsilon
        
        # Perturba t[i] por uma pequena quantidade epsilon e calcula a diferença
        t_eps = t.copy()
        t_eps[i] += epsilon
        J[:, N + i] = (f(w, t_eps, g, N) - f0) / epsilon
    
    return J

def metodo_newton(N, a, b, tol=1e-8, max_iter=100):
    """
    Resolve o sistema não linear pelo método de Newton para obter os pontos e pesos da quadratura de Gauss.
    """
    # Condições iniciais sugeridas para w e t (pesos e pontos)
    w = np.array([(b - a) / (2 * N) if i <= N/2 else (b - a) / (2 * N) for i in range(N)])
    
    # Calcula os pontos t (onde a função será avaliada)
    t = np.array([a + i * w[i] / 2 if i < N/2 else (a + b) - (a + (i - N//2) * w[i] / 2) for i in range(N)])
    
    # Caso N seja ímpar, ajusta o ponto do meio
    if N % 2 == 1:
        t[N//2] = (a + b) / 2
    
    # Calcula os valores g_j
    g = calcular_g(N, a, b)
    
    # Itera até que a convergência seja alcançada ou o número máximo de iterações seja atingido
    for _ in range(max_iter):
        f_val = f(w, t, g, N)  # Calcula o valor da função f(w, t)
        
        # Verifica se a solução atingiu a tolerância desejada
        if np.linalg.norm(f_val, np.inf) < tol:
            break
        
        # Calcula a Jacobiana e resolve o sistema linear
        J = jacobiana(w, t, g, N)
        s = la.solve(J, -f_val)
        
        # Atualiza os valores de w e t
        w += s[:N]
        t += s[N:]
    
    return t, w

def integrar_gauss(f, N, a, b):
    """
    Usa a quadratura de Gauss para integrar numericamente a função f no intervalo [a, b].
    """
    # Obtém os pontos e pesos de Gauss
    t, w = metodo_newton(N, a, b)
    
    # Calcula a integral usando a fórmula de quadratura de Gauss
    return np.sum(w * f(t))

def calcular_erro(f, N, a, b):
    """
    Calcula a integral exata, a integral aproximada usando Gauss e o erro absoluto.
    """
    # Calcula a integral exata usando o método de integração numérica de SciPy
    integral_exata, _ = integrate.quad(f, a, b)
    
    # Calcula a integral aproximada usando a quadratura de Gauss
    integral_aproximada = integrar_gauss(f, N, a, b)
    
    # Calcula o erro absoluto
    erro = abs(integral_exata - integral_aproximada)
    
    return integral_exata, integral_aproximada, erro


# Definição dos limites do intervalo de integração e o número de pontos de quadratura
a, b = 1, 3  # Intervalo de integração
N = 3  # Número de pontos de integração

# Calcula os pontos de integração e os pesos
pontos, pesos = metodo_newton(N, a, b)
print("Pontos de integração:", pontos)
print("Pesos:", pesos)

# Testando a integração numérica
funcao = lambda x: 3*np.exp(x)  # Exemplo de função

# Calculando erro
integral_exata, integral_aproximada, erro = calcular_erro(funcao, N, a, b)
print(f"Integral exata: {integral_exata}")
print(f"Integral aproximada: {integral_aproximada}")
print(f"Erro absoluto: {erro}")
