import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve

# Método de Newton para resolver sistemas não-lineares
def metodo_newton(f, J, w0, t0, tol=1e-8, max_iter=100):
    w, t = np.array(w0), np.array(t0)  # Condições iniciais para pesos e pontos
    for _ in range(max_iter):
        F = f(w, t)  # Avalia as equações não-lineares
        if np.linalg.norm(F, ord=np.inf) < tol:  # Verifica se a solução convergiu
            break
        J_matrix = J(w, t)  # Calcula a matriz Jacobiana
        try:
            delta = solve(J_matrix, -F)  # Resolve o sistema linear para encontrar o incremento
        except np.linalg.LinAlgError:
            break  # Se a matriz Jacobiana for singular, interrompe o processo
        w[:], t[:] = w + delta[:len(w)], t + delta[len(w):]  # Atualiza os valores de w e t
    return w, t

# Função para calcular os pesos e pontos de integração da quadratura de Gauss-Legendre
def gauss_legendre(N, a=-1, b=1):
    # Função para calcular os momentos (integrais de x^(j-1) no intervalo [a, b])
    def momentos(j):
        return quad(lambda x: x**(j-1), a, b)[0]
    
    # Vetor g contendo os momentos para j = 1, 2, ..., 2N
    g = np.array([momentos(j) for j in range(1, 2*N+1)])
    
    # Condições iniciais para os pesos e pontos
    w0 = np.full(N, (b - a) / (2 * (N + 1)))  # Inicialização dos pesos
    t0 = np.polynomial.legendre.leggauss(N)[0] * (b - a) / 2 + (a + b) / 2  # Chute inicial para os pontos
    
    # Função que define as equações não-lineares
    def equacoes(w, t):
        return np.array([sum(w[i] * t[i]**(j-1) for i in range(N)) - g[j-1] for j in range(1, 2*N+1)])
    
    # Função para calcular a matriz Jacobiana
    def jacobiano(w, t):
        epsilon = 1e-8  # Perturbação para calcular derivadas numéricas
        J = np.zeros((2*N, 2*N))  # Inicializa a matriz Jacobiana
        for j in range(2*N):
            for i in range(N):
                w_perturb = w.copy()
                t_perturb = t.copy()
                w_perturb[i] += epsilon  # Perturba o peso w[i]
                t_perturb[i] += epsilon  # Perturba o ponto t[i]
                # Calcula as derivadas parciais numericamente
                J[j, i] = (equacoes(w_perturb, t)[j] - equacoes(w, t)[j]) / epsilon
                J[j, N + i] = (equacoes(w, t_perturb)[j] - equacoes(w, t)[j]) / epsilon
        return J
    
    # Resolve o sistema não-linear usando o método de Newton
    w, t = metodo_newton(equacoes, jacobiano, w0, t0)
    return w, t

# Função para calcular a integral aproximada usando os pesos e pontos de integração
def integral_aproximada(f, w, t, a, b):
    return sum(w[i] * f(t[i]) for i in range(len(w)))

# Função para calcular a integral exata usando a função quad do scipy
def integral_exata(f, a, b):
    return quad(f, a, b)[0]

# Função para calcular o erro absoluto entre a integral aproximada e a exata
def calcular_erro(aproximada, exata):
    return abs(aproximada - exata)

def resultados(intervalos,f):
    lista_w = []
    lista_t = []
    intervalos_lista = []
    for a, b in intervalos:
        for N in range(1, 8):
            w, t = gauss_legendre(N, a, b)  # Calcula os pesos e pontos de integração
            lista_w.append(w)
            lista_t.append(t)
            aproximada = integral_aproximada(f, w, t, a, b)  # Calcula a integral aproximada
            exata = integral_exata(f, a, b)  # Calcula a integral exata
            erro = calcular_erro(aproximada, exata)  # Calcula o erro
            # Exibe os resultados
            intervalo = f"[{a}, {b}]"
            intervalos_lista.append(intervalo)
            print(intervalo,N,aproximada,exata,erro)
        print()
    return intervalos_lista,lista_w, lista_t

def imprimir_pesos_pontos(lista_i,lista_w, lista_t):
    n=1
    for i,w,t in zip(lista_i,lista_w, lista_t):
        print(i,n,w,t)
        n+=1
        if n==8:
            print()
            n=1
    print("\n")

f1 = lambda x: np.exp(x)
f2 = lambda x: x**2 - 3
f3 = lambda x: x*np.exp(x)

# Intervalos de integração
intervalos = [(-1, 1), (0, 2), (-2, 2)]

print("Intervalo, N, Integral aproximada, Integral exata, Erro")
print("e^x")
i1,w1,t1 = resultados(intervalos,f1)
print("x^2 - 3")
i2,w2,t2 = resultados(intervalos,f2)
print("x*e^x")   
i3,w3,t3 = resultados(intervalos,f3)

print("Intervalo, N, Pesos, Pontos")
print("e^x")
imprimir_pesos_pontos(i1,w1, t1)
print("x^2 - 3")
imprimir_pesos_pontos(i2,w2, t2)
print("x*e^x")
imprimir_pesos_pontos(i3,w3, t3)

'''
Os pesos w0 são inicializados uniformemente como w0 = (b - a) / (2 * (N + 1)),
garantindo que os valores sejam pequenos o suficiente para evitar grandes erros na iteração do método de Newton.
Já os pontos t0 são obtidos a partir dos zeros do polinômio de Legendre no intervalo padrão [-1,1]
usando np.polynomial.legendre.leggauss(N)[0], e são escalados para o intervalo desejado [a, b]
por meio da transformação t0 = ((b - a) / 2) * t + ((a + b) / 2),
garantindo que os cálculos sejam ajustados corretamente para qualquer
intervalo de integração. Essas escolhas permitem um ponto de partida
adequado para que o método de Newton encontre os pesos e pontos corretos da quadratura de Gauss-Legendre
de forma eficiente.

Em todos os casos estudados indique todos os dados utilizados nas simulacoes, como:
o intervalo [a, b], a tolerancia TOL, a perturbacao ε usada para aproximar a matriz
Jacobiana, o numero de particoes m da formula repetida de Newton-Cotes e numero
de pontos N.
TOL = 1e-8 - tolerância para o método de Newton
ε = 1e-8 - perturbação para calcular derivadas numéricas
m = 100 - número de partições para a fórmula repetida de Newton-Cotes'
'''