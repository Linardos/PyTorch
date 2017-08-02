#! usr/bin/python3.5
import numpy as np
import matplotlib.pyplot as plt
import os
from TorchPCA import PCA

if os.path.isfile('GDS6248.soft'):
	pass
else:
	os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
	os.system('gunzip GDS6248.soft.gz')
	os.system('tail -n +141 GDS6248.soft > GDS6248.softer') #getting rid of the redundant lines
	os.system('rm GDS6248.soft')
	os.system('head -n -1 GDS6248.softer > GDS6248.soft') #one last redundant line
	os.system('rm GDS6248.soft.gz')
	os.system('rm GDS6248.softer')

#In the following loop I'm keeping the float values while skipping the strings by setting the ValueError exception
temp = []
with open('GDS6248.soft') as f:
	for l in f:
		temp2=[]
		for x in l.split()[2:]:
			try:
				temp2.append(float(x))
			except ValueError:
				pass
		temp.append(temp2)

X=np.array(temp)
Color = ['m' for x in range(3)] + ['c' for x in range(24)] + ['r' for x in range(24)] #Color scheme for the samples


obj = PCA
obj.Decomposition(X,2)
obj.explained_variance()
