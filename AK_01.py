import math
import numpy as np
import matplotlib.pyplot as plt
import os
from graphviz import Digraph

# Graphviz PATH setzen (passe den Pfad an, falls anders installiert)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'


# ============================================================================
# TEIL 1: Numerische Differentiation - Grundlagen
# ============================================================================
# Zeigt wie man Ableitungen numerisch mit kleinen h-Werten approximiert

def beispiel_numerische_ableitung():
    """
    Demonstriert numerische Ableitung einer Funktion f(x) = 3x² - 4x + 5
    Ableitung: f'(x) = 6x - 4
    """
    def f(x):
        return 3*x**2 - 4*x + 5

    # Funktion plotten
    print("=== Numerische Ableitung ===")
    print(f"f(3.0) = {f(3.0)}")
    
    xs = np.arange(-5, 5, 0.25)
    ys = f(xs)
    plt.figure()
    plt.plot(xs, ys)
    plt.title("f(x) = 3x² - 4x + 5")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

    # Numerische Ableitung an Stelle x=3.0
    h = 0.001
    x = 3.0
    numerische_ableitung = (f(x + h) - f(x)) / h
    print(f"Numerische Ableitung bei x=3.0: {numerische_ableitung}")
    print(f"Analytische Ableitung bei x=3.0: {6*x - 4}")  # f'(x) = 6x - 4
    print()


def beispiel_partielle_ableitungen():
    """
    Berechnet partielle Ableitungen von d = a*b + c
    Zeigt wie sich d ändert, wenn a, b oder c leicht verändert wird
    """
    print("=== Partielle Ableitungen von d = a*b + c ===")
    h = 0.0001
    
    # Ableitung nach a: ∂d/∂a = b
    a, b, c = 2.0, -3.0, 10.0
    d1 = a*b + c
    d2 = (a + h)*b + c
    print(f"∂d/∂a = {(d2-d1)/h:.4f}  (erwartet: b = {b})")
    
    # Ableitung nach b: ∂d/∂b = a
    a, b, c = 2.0, -3.0, 10.0
    d1 = a*b + c
    d2 = a*(b + h) + c
    print(f"∂d/∂b = {(d2-d1)/h:.4f}  (erwartet: a = {a})")
    
    # Ableitung nach c: ∂d/∂c = 1
    a, b, c = 2.0, -3.0, 10.0
    d1 = a*b + c
    d2 = a*b + (c + h)
    print(f"∂d/∂c = {(d2-d1)/h:.4f}  (erwartet: 1)")
    print()


# ============================================================================
# TEIL 2: Value-Klasse - Computational Graph
# ============================================================================
# Implementiert einen Knoten im Computational Graph mit automatischer Differentiation

class Value:
    """
    Repräsentiert einen Wert im Computational Graph.
    Speichert den Wert (data), den Gradienten (grad) und die Verbindungen zu anderen Knoten.
    """
    
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data              # Der eigentliche Wert
        self.grad = 0.0               # Gradient (Ableitung) - wird später berechnet
        self._backward = lambda: None    # Funktion für Backpropagation (wird später definiert)
        self._prev = set(_children)   # Elternknoten (Inputs)
        self._op = _op                # Operation die diesen Knoten erzeugt hat (+, *, etc.)
        self.label = label            # Name für Visualisierung
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        """Addition: self + other"""
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        """Multiplikation: self * other"""
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out


# ============================================================================
# TEIL 3: Graph Visualisierung
# ============================================================================

def trace(root):
    """
    Durchläuft den Computational Graph und sammelt alle Knoten und Kanten.
    Startet bei root und geht rekursiv durch alle Elternknoten.
    """
    nodes, edges = set(), set()
    
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_dot(root):
    """
    Erstellt eine Graphviz-Visualisierung des Computational Graphs.
    Zeigt Werte (data) und Gradienten (grad) für jeden Knoten.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    
    # Knoten erstellen
    for n in nodes:
        uid = str(id(n))
        # Rechteckiger Knoten mit Label, Wert und Gradient
        dot.node(name=uid, 
                label="{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), 
                shape='record')
        
        # Wenn Knoten durch Operation entstanden ist, zeige Operation
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    # Kanten erstellen (Verbindungen zwischen Knoten)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot


# ============================================================================
# TEIL 4: Backpropagation - Manuelle Gradientenberechnung
# ============================================================================

def beispiel_backpropagation():
    """
    Demonstriert manuelle Backpropagation für L = (a*b + c) * f
    
    Forward Pass:
        e = a * b = 2.0 * (-3.0) = -6.0
        d = e + c = -6.0 + 10.0 = 4.0
        L = d * f = 4.0 * (-2.0) = -8.0
    
    Backward Pass (Chain Rule):
        ∂L/∂L = 1.0
        ∂L/∂f = d = 4.0
        ∂L/∂d = f = -2.0
        ∂L/∂c = ∂L/∂d * ∂d/∂c = -2.0 * 1 = -2.0
        ∂L/∂e = ∂L/∂d * ∂d/∂e = -2.0 * 1 = -2.0
        ∂L/∂a = ∂L/∂e * ∂e/∂a = -2.0 * b = -2.0 * (-3.0) = 6.0
        ∂L/∂b = ∂L/∂e * ∂e/∂b = -2.0 * a = -2.0 * 2.0 = -4.0
    """
    print("=== Backpropagation Beispiel ===")
    
    # Forward Pass: Berechne L = (a*b + c) * f
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f; L.label = "L"
    
    print(f"Forward Pass: L = {L.data}")
    
    # Backward Pass: Manuelle Gradientenberechnung
    L.grad = 1.0                    # ∂L/∂L = 1
    f.grad = d.data                 # ∂L/∂f = d
    d.grad = f.data                 # ∂L/∂d = f
    c.grad = d.grad * 1.0           # ∂L/∂c = ∂L/∂d * ∂d/∂c
    e.grad = d.grad * 1.0           # ∂L/∂e = ∂L/∂d * ∂d/∂e
    a.grad = e.grad * b.data        # ∂L/∂a = ∂L/∂e * ∂e/∂a
    b.grad = e.grad * a.data        # ∂L/∂b = ∂L/∂e * ∂e/∂b
    
    print(f"Gradienten: a={a.grad}, b={b.grad}, c={c.grad}, f={f.grad}")
    
    # Visualisierung mit Gradienten
    draw_dot(L).render('computation_graph', view=True)
    
    # Gradient Descent Schritt (Learning Rate = 0.01)
    learning_rate = 0.01
    a.data += learning_rate * a.grad
    b.data += learning_rate * b.grad
    c.data += learning_rate * c.grad
    f.data += learning_rate * f.grad
    
    # Neuer Forward Pass mit aktualisierten Werten
    e_new = a * b
    d_new = e_new + c
    L_new = d_new * f
    
    print(f"Nach Gradient Descent: L = {L_new.data}")
    print()


def numerische_verifikation():
    """
    Verifiziert die Gradienten numerisch mit h-Methode.
    Berechnet ∂L/∂a numerisch und vergleicht mit analytischem Gradient.
    """
    print("=== Numerische Verifikation der Gradienten ===")
    h = 0.0001
    
    # Original L
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f; L.label = "L"
    L1 = L.data
    
    # L mit leicht verändertem a
    a = Value(2.0 + h, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b; e.label = "e"
    d = e + c; d.label = "d"
    f = Value(-2.0, label="f")
    L = d * f; L.label = "L"
    L2 = L.data
    
    numerischer_gradient = (L2 - L1) / h
    analytischer_gradient = 6.0  # Aus Backpropagation bekannt
    
    print(f"Numerischer Gradient ∂L/∂a: {numerischer_gradient:.4f}")
    print(f"Analytischer Gradient ∂L/∂a: {analytischer_gradient:.4f}")
    print()

plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2))); plt.grid(); plt.show(); plt.close()

# inputs x1, x2
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
# weights w1, w2 and bias b
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")

x1w1 = x1*w1; x1w1.label = "x1*w1"
x2w2 = x2*w2; x2w2.label = "x2*w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1*w1 + x2*w2"
n = x1w1x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"

"""o.grad = 1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()"""

topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topt.append(v)
    build_topo(o)
topo.view()

draw_dot(o).view()
# ============================================================================
# MAIN - Führe alle Beispiele aus
# ============================================================================

if __name__ == "__main__":
    # Kommentiere aus, was du sehen möchtest:
    
    # beispiel_numerische_ableitung()
    # beispiel_partielle_ableitungen()
    beispiel_backpropagation()
    # numerische_verifikation()



