#!/usr/bin/env python3
"""Logistic regression — binary classification via gradient descent."""
import math

def sigmoid(z): return 1/(1+math.exp(-max(-500, min(500, z))))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000): self.lr=lr; self.epochs=epochs
    def fit(self, X, y):
        n, d = len(X), len(X[0]); self.w = [0]*d; self.b = 0
        for _ in range(self.epochs):
            for i in range(n):
                z = sum(self.w[j]*X[i][j] for j in range(d)) + self.b
                p = sigmoid(z); err = p - y[i]
                for j in range(d): self.w[j] -= self.lr * err * X[i][j] / n
                self.b -= self.lr * err / n
    def predict_proba(self, x):
        return sigmoid(sum(self.w[j]*x[j] for j in range(len(x))) + self.b)
    def predict(self, x): return 1 if self.predict_proba(x) >= 0.5 else 0

def main():
    X = [[0,0],[1,0],[0,1],[1,1],[2,2],[3,3]]; y = [0,0,0,0,1,1]
    lr = LogisticRegression(lr=0.5, epochs=500); lr.fit(X, y)
    for x in X: print(f"{x}: {lr.predict(x)} ({lr.predict_proba(x):.3f})")

if __name__ == "__main__": main()
