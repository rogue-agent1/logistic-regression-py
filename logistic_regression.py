#!/usr/bin/env python3
"""Logistic regression binary classifier."""
import math, random, sys

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000): self.lr=lr; self.epochs=epochs
    def _sigmoid(self, z): return 1/(1+math.exp(-max(-500,min(500,z))))
    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0]*d; self.b = 0.0
        for _ in range(self.epochs):
            for i in range(n):
                z = sum(X[i][j]*self.w[j] for j in range(d)) + self.b
                p = self._sigmoid(z); err = p - y[i]
                for j in range(d): self.w[j] -= self.lr*err*X[i][j]/n
                self.b -= self.lr*err/n
    def predict_proba(self, x):
        return self._sigmoid(sum(x[j]*self.w[j] for j in range(len(self.w))) + self.b)
    def predict(self, X): return [1 if self.predict_proba(x) > 0.5 else 0 for x in X]
    def accuracy(self, X, y): return sum(p==t for p,t in zip(self.predict(X),y))/len(y)

if __name__ == "__main__":
    random.seed(42)
    X = [[random.gauss(-2,1), random.gauss(-2,1)] for _ in range(30)] + \
        [[random.gauss(2,1), random.gauss(2,1)] for _ in range(30)]
    y = [0]*30 + [1]*30
    lr = LogisticRegression(lr=0.5, epochs=500); lr.fit(X, y)
    print(f"Accuracy: {lr.accuracy(X, y):.1%}")
    print(f"Weights: [{lr.w[0]:.3f}, {lr.w[1]:.3f}], Bias: {lr.b:.3f}")
