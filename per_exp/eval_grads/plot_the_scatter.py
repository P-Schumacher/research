import numpy as np
from pudb import set_trace
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import numpy as np
from scipy import stats
import scipy.stats, numpy

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
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

y = np.load('scatter_metric.npy')
x = np.load('scatter_td_error.npy')
x = x[:, 0]

xmin = x.min()
xmax = x.max()
ymin = y.min()
ymax = y.max()
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x, y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
#plt.imshow(np.rot90(Z), cmap=plt.cm.hot,
#          extent=[xmin, xmax, ymin, ymax])
#plt.plot(x, y, 'k.', markersize=1, marker='o')
#cb = plt.colorbar()
#cb.set_ticks([np.min(Z), (np.max(Z)+np.min(Z))/2, np.max(Z)])
#cb.set_ticklabels([0, 0.5, 1])


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

# set-up the plot
#plt.axes().set_aspect('equal')
plt.xlabel('TD-error')
plt.ylabel('Cos. Sim. with hq-critic')
plt.title('Linear regression and confidence limits')
# plot sample data
plt.plot(x,y,'bo', label='Sample observations', markersize=4)
# plot line of best fit
plt.plot(p_x, p_y, 'r--', label='P(x) = ax + b')
plt.fill_between(p_x, lower, upper, color='r', alpha=0.2, label='CI 95%')# label='Lower confidence limit (95%)')
plt.xlim([np.min(x), np.max(x)])
# configure legend
plt.legend(loc=0)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=10)

# show the plot
r, p, lo, hi = pearsonr_ci(x, y)
print(r)
print(p)
print(lo)
print(hi)
plt.show()


