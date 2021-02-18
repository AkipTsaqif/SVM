# SRI AHMAD TSAQIF
# AKIP TSAQIF

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC

# Functions to find margins
def geo_margin(y, w, b, x):
    return y * (w / np.linalg.norm(w)).T @ x + (b / np.linalg.norm(w)).flatten()

def functional_margin(y, w, b, x):
    return y * (w.T @ x + b).flatten()

def f(x, w, b, c=0):
    return (-w[0] * x - b + c) / w[1]

# Inputting the data
xneg = np.array([[1,4], [2,3], [3,4]])
xpos = np.array([[5,-3], [6,-1], [7,-1]])
yneg = np.array([-1, -1, -1])
ypos = np.array([1, 1, 1])

x1 = np.linspace(-10, 10)
x = np.vstack((np.linspace(-10, 10), np.linspace(-10, 10)))

# Guessing w and b first
w = np.array([1, -1]).reshape(-1, 1)
b = -3

# Drawing the graph
fig = plt.figure(figsize = (10,10))
plt.scatter(xneg[:,0], xneg[:,1], marker = 'x', color = 'r', label = 'Negative Y')
plt.scatter(xpos[:,0], xpos[:,1], marker = 'o', color = 'b',label = 'Positive Y')
plt.plot(x1, x1  - 3, color = 'darkblue')
plt.plot(x1, x1  - 7, linestyle = '--', alpha = .3, color = 'b')
plt.plot(x1, x1  + 1, linestyle = '--', alpha = .3, color = 'r')
plt.xlim(0,10)
plt.ylim(-5,5)
plt.xticks(np.arange(0, 10, step=1))
plt.yticks(np.arange(-5, 5, step=1))

# Lines
plt.axvline(0, color = 'black', alpha = .5)
plt.axhline(0,color = 'black', alpha = .5)
plt.plot([2,6],[3,-1], linestyle = '-', color = 'darkblue', alpha = .5 )
plt.plot([4,6],[1,1],[6,6],[1,-1], linestyle = ':', color = 'darkblue', alpha = .5 )
plt.plot([0,1.5],[0,-1.5],[6,6],[1,-1], linestyle = ':', color = 'darkblue', alpha = .5 )

# Annotations
plt.annotate(text = '$A \ (6,-1)$', xy = (6,-1), xytext = (5.5,-1.4), fontsize = 15)
plt.annotate(text = '$B \ (2,3)$', xy = (2,3), xytext = (1.5,3.2), fontsize = 15)
plt.annotate(text = '$2$', xy = (5,1.2), xytext = (5,1.2), fontsize = 12)
plt.annotate(text = '$2$', xy = (6.2,.5), xytext = (6.2,.5), fontsize = 12)
plt.annotate(text = '$2\sqrt{2}$', xy = (4.5,-.5), xytext = (4.5,-.5), fontsize = 13)
plt.annotate(text = '$2\sqrt{2}$', xy = (2.5,1.5), xytext = (2.5,1.5), fontsize = 13)
plt.annotate(text = '$w^Tx + b = 0$', xy = (8,4.5), xytext = (7.3,3.8), fontsize = 17)
plt.annotate(text = '$(\\frac{1}{4},-\\frac{1}{4}) \\binom{x_1}{x_2}- \\frac{3}{4} = 0$', xy = (7.5,4), xytext = (6.7,3.2), fontsize = 19)
plt.annotate(text = '$\\frac{3}{\sqrt{2}}$', xy = (.5,-1), xytext = (.4,-1.1), fontsize = 16)
plt.annotate(text = 'SRI AHMAD TSAQIF', xy = (7.2,-3.5), fontsize = 15)
plt.annotate(text = 'Made with PyPlot', xy = (7.2,-3.8), fontsize = 13)

# Labels and show
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower right')
plt.show()

# Inputting back the data to find its true coefficient (W) and bias (B)
X = np.array([[1,4], [2,3], [3,4], [5,-3], [6,-1], [7,-1]])
y = np.array([-1, -1, -1, 1, 1, 1])

clf = SVC(C = 1, kernel = 'linear')
clf.fit(X, y)

# Printing the results
print('w =', clf.coef_)
print('b =', clf.intercept_)
print('Indices of support vectors =', clf.support_)
print('Support vectors =\n', clf.support_vectors_)
print('Number of support vectors =', clf.n_support_)
print('Coefficients of the support vector in the decision function =', np.abs(clf.dual_coef_))

print('\n')
print('Wx + b = 0\nInputting (1, 4) into the equation gives: -1.5')
print('-1.5 < 0, hence the final result for x1 (1, 4) is: ', clf.predict([[1,4]]))
print()
print('Wx + b = 0\nInputting (5, -3) into the equation gives: 1.25')
print('1.25 > 0, hence the final result for x1 (5, -3) is: ', clf.predict([[5,-3]]))
