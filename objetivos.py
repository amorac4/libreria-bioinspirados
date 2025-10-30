# objetivos.py
from __future__ import annotations
import numpy as np
from typing import Callable

# Funciones n-dimensionales
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

# Funciones 2D
def himmelblau(x):
    """Función de Himmelblau (solo 2D)."""
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return (a*a + b - 11)**2 + (a + b*b - 7)**2

def rosenbrock(x, a=1, b=100):
    """
    Función 'Banana' de Rosenbrock.
    Mínimo global f(a,a,...,a) = 0.
    """
    x = np.asarray(x)
    
    # Comprueba si la entrada es un solo vector (1D)
    is_1d = (x.ndim == 1)
    
    # Si es 1D, la convertimos temporalmente a 2D (con 1 fila)
    if is_1d:
        x = x[np.newaxis, :]
        
    # Cálculo vectorizado (funciona para 1 o N vectores)
    term1 = b * (x[..., 1:] - x[..., :-1]**2)**2
    term2 = (a - x[..., :-1])**2
    
    res = np.sum(term1 + term2, axis=-1) # Suma a lo largo de las dimensiones
    
    return res.item() if is_1d else res

def beale(x):
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return (1.5 - a + a*b)**2 + (2.25 - a + a*b*b)**2 + (2.625 - a + a*b**3)**2

def booth(x):
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return (a + 2*b - 7)**2 + (2*a + b - 5)**2

def camel3(x):
    x = np.asarray(x)
    a, b = x[..., 0], x[..., 1]
    return 2*a*a - 1.05*a**4 + (a**6)/6 + a*b + b*b



def griewank(x):

    x = np.asarray(x)
    n = x.shape[-1]
    
    sum_term = np.sum(x**2 / 4000, axis=-1)
    
    # Crea un array [sqrt(1), sqrt(2), ..., sqrt(n)]
    i = np.arange(1, n + 1)
    cos_term = np.cos(x / np.sqrt(i))
    prod_term = np.prod(cos_term, axis=-1)
    
    return sum_term - prod_term + 1


def schwefel(x):

    x = np.asarray(x)
    n = x.shape[-1]
    
    sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)
    
    return 418.9829 * n - sum_term


def easom(x):

    x = np.asarray(x)
    a = x[..., 0]
    b = x[..., 1]
    
    term1 = -np.cos(a) * np.cos(b)
    term2 = np.exp(-((a - np.pi)**2 + (b - np.pi)**2))
    
    return term1 * term2

# Diccionario para que main.py encuentre las funciones por su nombre
OBJETIVOS_REGISTRADOS: dict[str, Callable] = {
    "sphere": sphere,
    "ackley": ackley,
    "rastrigin": rastrigin,
    "himmelblau": himmelblau,
    "rosenbrock": rosenbrock,
    "beale": beale,
    "booth": booth,
    "camel3": camel3,
    "griewank": griewank,
    "schwefel": schwefel,
    "easom": easom,
}