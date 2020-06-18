import numpy as np
from pudb import set_trace
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import numpy as np
from scipy import stats
import scipy.stats, numpy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.style.use('seaborn')
plt.figure(figsize=[5,5])
def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

x_samples = np.load('scatter_td_error_0_1000.npy')
y_samples = np.load('scatter_metric_0_1000.npy')

x = []
for x1 in x_samples:
    for x2 in x1:
        x.append(x2) 
y = []
for y1 in y_samples:
    for y2 in y1:
        y.append(y2) 
x, y = np.asarray(x), np.asarray(y)
    
r, p, lo, hi = pearsonr_ci(x, y)
print(r)
print(p)
print(lo)
print(hi)

# fit a curve to the data using a least squares 1st order polynomial fit
z = np.polyfit(x,y,1)
p = np.poly1d(z)
fit = p(x)

# get the coordinates for the fit curve
c_y = [np.min(fit),np.max(fit)]
c_x = [np.min(x),np.max(x)]

# predict y values of origional data using the fit
p_y = z[0] * x + z[1]

# calculate the y-error (residuals)
y_err = y - p_y

# create series of new test x-values to predict for
p_x = np.arange(np.min(x),np.max(x)+1,1)

# now calculate confidence intervals for new test x-series
mean_x = np.mean(x)         # mean of x
n = len(x)              # number of samples in origional fit
t = 2.31                # appropriate t value (where n=9, two tailed 95%)
s_err = np.sum(np.power(y_err,2))   # sum of the squares of the residuals

confs = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((p_x-mean_x),2)/
            ((np.sum(np.power(x,2)))-n*(np.power(mean_x,2))))))

# now predict y based on test x-values
p_y = z[0] * p_x + z[1]

# get lower and upper confidence limits based on predicted y and confidence intervals
lower = p_y - abs(confs)
upper = p_y + abs(confs)

for i in range(x_samples.shape[0]):
    plt.plot(x_samples[i, :], y_samples[i, :], 'o', markersize=5, label=f'Seed {i}')
plt.plot(p_x, p_y, 'r--', label='P(x) = ax + b')
plt.fill_between(p_x, lower, upper, color='r', alpha=0.2, label='CI 95%')

plt.xlabel('TD-error')
#plt.ylabel('Cos. Sim. with hq-critic')
plt.ylabel('d$_{\cos}(G_{\mathrm{lq}}, G_{\mathrm{hq}})$')
plt.xlim([np.min(x), np.max(x)])
# configure legend
plt.legend(loc=0, frameon=True)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=10)
plt.title('$k\' / k$: 0')
plt.savefig('scatterplot_0_1000.pdf')
plt.show()
