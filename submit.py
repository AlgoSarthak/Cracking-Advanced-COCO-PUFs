import numpy as np
import sklearn.svm
from scipy.linalg import khatri_rao

def my_map( X ):

	feat = np.empty((X.shape[0], 2*X.shape[1]-1))
	for i in range(X.shape[0]):

		d = 1 - 2 * X[i, :31]
		feat[i, :31] = d
		d = np.append(d, 1 - 2*X[i, 31])
		cumulative_products = np.cumprod(d[::-1])[::-1]
		feat[i, 31:] = cumulative_products

	return feat

def my_fit( X_train, y0_train, y1_train ):

  model = sklearn.svm.LinearSVC(C=100, tol=1e-5, penalty='l2', max_iter=1000, loss='squared_hinge', dual=False)
  feat = my_map(X_train)
  model.fit(feat, y0_train)
  w0 = model.coef_
  b0 = model.intercept_

  model.fit(feat, y1_train)
  w1 = model.coef_
  b1 = model.intercept_
  return w0, b0, w1, b1
