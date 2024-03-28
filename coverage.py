import numpy as np
import matplotlib.pyplot as plt

def simulation(theta):
    return theta**(-1.5) + np.random.normal(0, 0.001)

def prior(N):
    return np.random.uniform(1, 10, N)

def likelihood(theta, data):
    return -0.5 * (data - theta**(-1.5))**2/0.1**2 - 0.5*np.log(0.001**2)

def posterior(theta, data):
    return likelihood(theta, data) + np.log(1/9)

N = 10000
samples = prior(N)
data = [simulation(samples[i]) for i in range(N)]

true_data = simulation(3)
post = np.array([posterior(samples[i], true_data) for i in range(N)])
post = np.exp(post - np.max(post)) # posterior estimator

plt.hist(samples, bins=50, density=True, alpha=0.5, label='Prior')
plt.hist(samples, weights=post, 
         bins=50, density=True, alpha=0.5, label='Posterior')
plt.legend()
plt.show()

alpha_hpdr = []
for i in range(len(samples)):
    alpha = np.mean([1])