import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulator(theta):
    return theta**2 + np.random.normal(0, 0.5, 1)

def prior(N):
    return np.random.uniform(1, 10, N)

def posterior(theta, data):
    return norm.pdf(theta**2, loc=data, scale=0.5)

def generate_samples(n, data):
    return norm.rvs(loc=data, scale=0.1, size=n)

N = 1000
samples = prior(N)
samples = np.sort(samples)
true_data = 2.3**2 + np.random.normal(0, 0.5, 1)

post = np.array([posterior(samples[i], true_data) for i in range(N)])
post = post/np.max(post) # posterior estimator
print(post)

plt.hist(samples, bins=50, density=True, alpha=0.5, label='Prior')
plt.hist(samples, weights=post, 
         bins=50, density=True, alpha=0.5, label='Posterior')
plt.axvline(2.3, ls='--')
plt.legend()
plt.show()
sys.exit(1)

test_samples = generate_samples(1000)
f = []
for i in range(len(test_samples)):
    f = 0