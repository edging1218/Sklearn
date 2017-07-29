import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
data_size = 50
# y = ax + b + epsilon
np.random.seed(1234)
x = np.random.normal(0, 2, data_size)
epsilon = np.random.normal(0, 1, data_size)

#plot the distribution function of x
sns.distplot(x)
plt.title('Distribution of X')
plt.show()

a = 2
b = 1
y = a * x + b + epsilon
lm = pd.DataFrame({'x':x, 'y': y})

#linear regression of x and y
sns.lmplot(x = 'x', y = 'y', data= lm)
plt.title('Linear regression')
plt.show()

#plot residuals
sns.residplot(x='x', y='y', data=lm, color='green')
plt.title('Residuals')
plt.show()

#plot polynomial regression
c = 3
y2 = a * x + b + epsilon + c * np.square(x)
lm['y2'] = y2
sns.regplot(x = 'x', y = 'y2', data= lm, color = 'blue', scatter = True, label = 'order 1')
sns.regplot(x = 'x', y = 'y2', data= lm, color = 'green', scatter = None, label = 'order 2', order = 2)
plt.title('Polynomial regression')
plt.legend(loc = 'upper right')
plt.show()


#plot by category
lm['category'] = np.random.randint(3, size = data_size)
sns.lmplot(x = 'x', y = 'y', data= lm, hue = 'category', palette = 'Set1')
plt.title('Plot data by groups')
plt.show()


#Plot by category (separate in rows)
sns.lmplot(x = 'x', y = 'y', data= lm, row = 'category')
plt.show()


#Strip plot
plt.subplot(2,1,1)
sns.stripplot(x='category', y='y', data=lm)

# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x='category', y='y', data=lm, jitter=True, size=3)

plt.show()

#Jointplot
sns.jointplot(x='x', y='y', data=lm)
plt.show()

sns.jointplot(x='x', y='y', data=lm, kind = 'hex')
plt.show()

#Pairwise joint distribution
sns.pairplot(lm, vars=['x', 'y'])
plt.show()

sns.pairplot(lm, vars = ['x', 'y'], hue='category',kind='reg')
plt.show()

#Covariance matrix
cov_m = np.cov(np.vstack((x, y, y2)))
sns.heatmap(cov_m)
plt.show()

