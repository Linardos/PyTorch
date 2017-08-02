import torch
import numpy as np
import matplotlib.pyplot as plt

class PCA:
	def __init__(self, Data):
		self.Data = Data

	def __repr__(self):
		return f'PCA({self.Data})'

	@staticmethod
	def Center(Data):
		#Convert to torch Tensor and keep the number of rows and columns
		t = torch.from_numpy(Data)
		no_rows, no_columns = t.size()
		row_means = torch.mean(t, 1)
		#Expand the matrix in order to have the same shape as X and substract, to center
		for_subtraction = row_means.expand(no_rows, no_columns)
		X = t - for_subtraction #centered
		return(X)

	@classmethod
	def Decomposition(cls, Data, k):
		#Center the Data using the static method within the class
		X = cls.Center(Data)
		U,S,V = torch.svd(X)
		eigvecs=U.t()[:,:k] #the first k vectors will be kept
		y=torch.mm(U,eigvecs)

		#Save variables to the class object, the eigenpair and the centered data
		cls.eigenpair = (eigvecs, S)
		cls.data=X
		return(y)

	def explained_variance():
		#Total sum of eigenvalues (total variance explained)
		tot = sum(PCA.eigenpair[1])
		#Variance explained by each principal component
		var_exp = [(i / tot) for i in sorted(PCA.eigenpair[1], reverse=True)]
		cum_var_exp = np.cumsum(var_exp)
		#X is the centered data
		X = PCA.data
		#Plot both the individual variance explained and the cumulative:
		plt.bar(range(X.size()[1]), var_exp, alpha=0.5, align='center', label='individual explained variance')
		plt.step(range(X.size()[1]), cum_var_exp, where='mid', label='cumulative explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.show()




def KPCA(X, gamma=3, dims=1, mode='gaussian'):
	print('Now running Kernel PCA with', mode, 'kernel function...')
	'''
	X is the necessary input. The data.
	gamma will be the user defined value that will be used in the kernel functions. The default is 3.
	dims will be the number of dimensions of the final output (basically the number of components to be picked). The default is 1.
	mode has three options 'gaussian', 'polynomial', 'hyperbolic tangent' which will be the kernel function to be used. The default is gaussian.
	'''

	#First the kernel function picked by the user is defined. Vectors need to be input in np.mat type

	def phi(x1,x2):
		if mode == 'gaussian':
			return (float(np.exp(-gamma*((x1-x2).dot((x1-x2).T))))) #gaussian. (vectors are rather inconvenient in python, so instead of xTx for inner product we need to calculate xxT)
		if mode == 'polynomial':
			return (float((1 + x1.dot(x2.T))**gamma)) #polynomial
		if mode == 'hyperbolic tangent':
			return (float(np.tanh(x1.dot(x2.T) + gamma))) #hyperbolic tangent
	Kernel=[]
	for x in X.T:
		xi=np.mat(x)
		row=[]
		for y in X.T:
			xj=np.mat(y)
			kf=phi(xi,xj)
			row.append(kf)
		Kernel.append(row)
	kernel=np.array(Kernel)

	# Centering the symmetric NxN kernel matrix.
	N = kernel.shape[0]
	one_n = np.ones((N,N)) / N
	kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n) #centering

	eigVals, eigVecs = linalg.eigh(kernel) #the eigvecs are sorted in ascending eigenvalue order.
	y=eigVecs[:,-dims:].T #user defined dims, since the order is reversed, we pick principal components from the last columns instead of the first
	return (y)
