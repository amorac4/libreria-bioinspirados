# objetivos.py
from __future__ import annotations
import numpy as np
from typing import Callable

def sphere(x):
    """Función Esfera. Simple. Mínimo global f(0,0,...,0) = 0."""
    x = np.asarray(x)
    return np.sum(x*x, axis=-1)

def ackley(x, a=20, b=0.2, c=2*np.pi):
    """Función Ackley. Compleja, con muchos mínimos locales."""
    x = np.asarray(x)
    n = x.shape[-1]
    s1 = np.sum(x*x, axis=-1)
    s2 = np.sum(np.cos(c*x), axis=-1)
    return -a * np.exp(-b * np.sqrt(s1/n)) - np.exp(s2/n) + a + np.e

def rastrigin(x, A=10):
    """Función Rastrigin. Famosa por tener un "paisaje" muy rugoso."""
    x = np.asarray(x)
    n = x.shape[-1]
    return A*n + np.sum(x*x - A * np.cos(2*np.pi*x), axis=-1)

def himmelblau(x):
    """Función de Himmelblau (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return (a*a + b - 11)**2 + (a + b*b - 7)**2

def rosenbrock(x, a=1, b=100):
    """Función de Rosenbrock. Unimodal, con un valle estrecho."""
    x = np.asarray(x)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x[np.newaxis, :]
    term1 = b * (x[..., 1:] - x[..., :-1]**2)**2
    term2 = (a - x[..., :-1])**2
    res = np.sum(term1 + term2, axis=-1) 
    return res.item() if is_1d else res

def beale(x):
    """Función de Beale (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return (1.5 - a + a*b)**2 + (2.25 - a + a*b*b)**2 + (2.625 - a + a*b**3)**2

def booth(x):
    """Función de Booth (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return (a + 2*b - 7)**2 + (2*a + b - 5)**2

def camel3(x):
    """Función de Camel 3 (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return 2*a*a - 1.05*a**4 + (a**6)/6 + a*b + b*b

def michaelwicz2d(x, m=10):
    """Función de Michaelwicz (solo 2D)."""
    x = np.asarray(x)
    x1 = x[..., 0]
    x2 = x[..., 1] 
    term1 = np.sin(x1) * (np.sin((1 * x1**2) / np.pi))**(2*m)
    term2 = np.sin(x2) * (np.sin((2 * x2**2) / np.pi))**(2*m)  
    return -(term1 + term2)

def griewank(x):
    """Función de Griewank. Multimodal, con muchos mínimos locales."""
    x = np.asarray(x)
    n = x.shape[-1]
    sum_term = np.sum(x**2 / 4000, axis=-1)
    i = np.arange(1, n + 1)
    cos_term = np.cos(x / np.sqrt(i))
    prod_term = np.prod(cos_term, axis=-1)
    return sum_term - prod_term + 1


def schwefel(x):
    """Función de Schwefel. Multimodal, con muchos mínimos locales."""
    x = np.asarray(x)
    n = x.shape[-1]
    sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)
    return 418.9829 * n - sum_term


def easom(x):
    """Función de Easom (solo 2D)."""
    x = np.asarray(x)
    a = x[..., 0]
    b = x[..., 1]
    term1 = -np.cos(a) * np.cos(b)
    term2 = np.exp(-((a - np.pi)**2 + (b - np.pi)**2))
    
    return term1 * term2

def sum_squares(x):
    """Función Sum Squares (Elipsoidal). Unimodal. Mínimo global f(0)=0.
    f(x) = sum_{i=1}^{n} i * x_i^2
    """
    x = np.asarray(x)
    n = x.shape[-1]
    weights = np.arange(1, n + 1)
    return np.sum(weights * x**2, axis=-1)


def weierstrass(x, a=0.5, b=3, k_max=20):
    """Función de Weierstrass. Multimodal, muy difícil. Mínimo global f(0,0,...,0) = 0.
    Parámetros estándar: a=0.5, b=3, k_max=20.
    """
    x = np.asarray(x)
    is_1d = (x.ndim == 1)
    if is_1d:
        x = x[np.newaxis, :]
    n = x.shape[-1]
    k = np.arange(k_max + 1).reshape(-1, 1, 1)
    x_broadcasted = x[np.newaxis, :, :]
    outer_sum_terms = a**k * np.cos(2 * np.pi * b**k * (x_broadcasted + 0.5))
    inner_sum = np.sum(outer_sum_terms, axis=0)
    sum_over_n = np.sum(inner_sum, axis=-1)
    bias = n * np.sum(a**np.arange(k_max + 1) * np.cos(np.pi * b**np.arange(k_max + 1)))
    return (sum_over_n - bias).item() if is_1d else sum_over_n - bias

def levy(x):
    """Función de Lévy. Multimodal, con muchos mínimos locales."""
    x = np.asarray(x)
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[..., 0])**2
    term3 = (w[..., -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[..., -1])**2)
    k = np.arange(x.shape[-1] - 1)
    wi = w[..., :-1]
    term2 = np.sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2), axis=-1)
    return term1 + term2 + term3

def zakharov(x):
    """Función de Zakharov. Unimodal, con un valle estrecho."""
    x = np.asarray(x)
    sum1 = np.sum(x**2, axis=-1)
    n = x.shape[-1]
    i = np.arange(1, n + 1)
    sum2 = np.sum(0.5 * i * x, axis=-1)
    return sum1 + sum2**2 + sum2**4

def bukin6(x):
    """Función de Bukin N.6 (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    term1 = 100 * np.sqrt(np.abs(b - 0.01 * a**2))
    term2 = 0.01 * np.abs(a + 10)
    return term1 + term2

def matyas(x):
    """Función de Matyas (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return 0.26 * (a**2 + b**2) - 0.48 * a * b

def threehumpcamel(x):
    """Función de Three-Hump Camel (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return 2*a**2 - 1.05*a**4 + (a**6)/6 + a*b + b**2

def michaelwicz(x, m=10):
    """Función de Michaelwicz (N-Dimensional)."""
    x = np.asarray(x)
    i = np.arange(1, x.shape[-1] + 1)
    term = np.sin(x) * (np.sin(i * x**2 / np.pi))**(2 * m)
    return -np.sum(term, axis=-1)







# Diccionario de Funciones (Callable)
OBJETIVOS_REGISTRADOS = {
    "sphere": sphere,
    "ackley": ackley,
    "rastrigin": rastrigin,
    "griewank": griewank,
    "rosenbrock": rosenbrock,
    "schwefel": schwefel,
    "himmelblau": himmelblau,
    "beale": beale,
    "booth": booth,
    "camel3": camel3,
    "easom": easom,
    "michaelwicz2d": michaelwicz2d,
    "sum_squares": sum_squares,
    "weierstrass": weierstrass,
    "levy": levy,
    "zakharov": zakharov,
    "bukin6": bukin6,
    "matyas": matyas,
    "threehumpcamel": threehumpcamel,
    "michaelwicz": michaelwicz,
}

# Diccionario de Metadatos (Límites y Dimensiones Estándar)
INFO_OBJETIVOS = {
    # N-Dimensionales 
    "sphere":      {"bounds": [-100, 100],   "default_dim": 30},
    "ackley":      {"bounds": [-32, 32],     "default_dim": 30},
    "rastrigin":   {"bounds": [-5.12, 5.12], "default_dim": 30},
    "griewank":    {"bounds": [-600, 600],   "default_dim": 30},
    "rosenbrock":  {"bounds": [-30, 30],     "default_dim": 30}, 
    "schwefel":    {"bounds": [-500, 500],   "default_dim": 30},
    "sum_squares": {"bounds": [-10, 10],      "default_dim": 30},
    "weierstrass": {"bounds": [-0.5, 0.5],    "default_dim": 30},
    "levy":        {"bounds": [-10, 10],     "default_dim": 30},
    "zakharov":    {"bounds": [-5, 10],      "default_dim": 30},
    "michaelwicz": {"bounds": [0, np.pi],    "default_dim": 30}, # Aprox [0, 3.14]
    #


    # 2D Fijas 
    "himmelblau":    {"bounds": [-5, 5],          "fixed_dim": 2},
    "beale":         {"bounds": [-4.5, 4.5],      "fixed_dim": 2},
    "booth":         {"bounds": [-10, 10],        "fixed_dim": 2},
    "camel3":        {"bounds": [-5, 5],          "fixed_dim": 2},
    "easom":         {"bounds": [-100, 100],      "fixed_dim": 2},
<<<<<<< HEAD
    "michaelwicz2d": {"bounds": [0, np.pi],       "fixed_dim": 2},
    "bukin6":       {"bounds": [-15, 5],         "fixed_dim": 2},
    "matyas":       {"bounds": [-10, 10],        "fixed_dim": 2},
    "threehumpcamel":{"bounds": [-5, 5],         "fixed_dim": 2},

=======
    "michaelwicz2d": {"bounds": [0, np.pi],       "fixed_dim": 2}, 
>>>>>>> d010362d436a0a841153f3326d2f5b6735baad59
}