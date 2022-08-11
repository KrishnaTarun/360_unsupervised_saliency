from functools import partial
import numpy as np
from numpy import random
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import re, os, glob
from sklearn.mixture import GaussianMixture

EPSILON = np.finfo('float').eps

#### METRICS --
'''
Commonly used metrics for evaluating saliency map performance.

Most metrics are ported from Matlab implementation provided by http://saliency.mit.edu/
Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.

Python implementation: Chencan Qian, Sep 2014

[1] Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.
[repo] https://github.com/herrlich10/saliency
'''
def normalize(x, method='standard', axis=None):
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res

def visualize(gt,pred):
	gt = gt / gt.max()
	pred = pred/pred.max()
	gt = np.reshape(gt,gt.shape[0]*gt.shape[1])
	gt = np.array([gt])
	gt = np.transpose(gt)
	pred = np.reshape(pred,pred.shape[0]*pred.shape[1])
	pred = np.array([pred])
	pred = np.transpose(pred)
	gm = GaussianMixture(n_components=20).fit(gt)
	GT = gm.score_samples(gt)
	GT = np.absolute(GT)
	gm = GaussianMixture(n_components=20).fit(pred)
	P = gm.score_samples(pred)
	P = np.absolute(P)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(GT/GT.max(), '-k', color='red')
	ax.plot(P/P.max(), '-k')
	plt.legend(['GT','Pred'])
	plt.show()
	a,b,c = plt.hist(GT/GT.max(),bins=200)
	d = filters.gaussian_filter(a,1,0)
	n,m,o = plt.hist(P/P.max(),bins=200)
	l = filters.gaussian_filter(n,1,0)
	plt.plot(b[:200],d/d.max())
	plt.plot(m[:200],l/l.max(),color='red')
	plt.show()
	plt.plot(b[:200],d/d.max())
	plt.plot(m[:200],l/l.max(),color='red')
	plt.legend(['Ground Truth','Prediction'])
	plt.show()

def match_hist(image, cdf, bin_centers, nbins=256):
	image = img_as_float(image)
	old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
	new_bin = np.interp(old_cdf, cdf, bin_centers)
	out = np.interp(image.ravel(), old_bin, new_bin)
	return out.reshape(image.shape)

	
def JSD_Bernouli(p, q):
	t1 = p * np.log((2*p)/(q+p+EPSILON)+EPSILON)
	t2 =  q * np.log((2*q) / (q+p+EPSILON)+EPSILON)
	t3 = (p-1) * np.log((2*(p-1)) / (q+p-2+EPSILON)+EPSILON)
	t4 = (q-1) * np.log((2*(q-1)) / (q+p-2+EPSILON)+EPSILON)
	Jssd = (t1+t2-t3-t4)/2
	cv2.imwrite('./Visualize/'+img[9:11]+'_JSD_Ber.png',normalize(Jssd, method='range')*255)
	return 100*Jssd.mean()

def KLD_Bernouli(p,q):
	t1 = p*np.log(p/(q+EPSILON)+EPSILON)
	t2 = (1-p)*np.log((1-p)/(1-q+EPSILON)+EPSILON)
	kld = t1+t2
	kld = normalize(kld, method='range')
	cv2.imwrite('./Visualize/'+img[9:11]+'_KLD_Ber.png',kld*255)
	return 10*np.mean(kld)


def JSD(p,q):
	p = normalize(p, method='sum')
	q = normalize(q, method='sum')
	jsd = 0.5*(KL(p, 0.5*(p+q))+KL(q, 0.5*(p+q)))
	
	jsd = normalize(jsd, method='range')
	cv2.imwrite('./Visualize/'+img[9:11]+'_JSD.png',jsd*255)
	return 0.5*(KLD(p, 0.5*(p+q))+KLD(q, 0.5*(p+q)))

def KL(p, q):
	p = normalize(p, method='sum')
	q = normalize(q, method='sum')
	kld = np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0)

	return kld


def KLD(p, q):
	p = normalize(p, method='sum')
	q = normalize(q, method='sum')
	kld = np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0)
	kld = normalize(kld, method='range')
	cv2.imwrite('./Visualize/'+img[9:11]+'_KLD.png',kld*255)
	return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))

def AUC_Judd(saliency_map, fixation_map, jitter=False):
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Jitter the saliency map slightly to disrupt ties of the same saliency value
	if jitter:
		saliency_map += random.rand(*saliency_map.shape) * 1e-7
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# Calculate AUC
	thresholds = sorted(S_fix, reverse=True)
	tp = np.zeros(len(thresholds)+2)
	fp = np.zeros(len(thresholds)+2)
	tp[0] = 0; tp[-1] = 1
	fp[0] = 0; fp[-1] = 1
	for k, thresh in enumerate(thresholds):
		above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
		tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
		fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
	return np.trapz(tp, fp) # y, x

def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# For each fixation, sample n_rep values from anywhere on the sasalmap2liency map
	if rand_sampler is None:
		r = random.randint(0, n_pixels, [n_fix, n_rep])
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
	else:
		S_rand = rand_sampler(S, F, n_rep, n_fix)
	# Calculate AUC per random split (set of random locations)
	auc = np.zeros(n_rep) * np.nan
	for rep in range(n_rep):
		thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
		tp = np.zeros(len(thresholds)+2)
		fp = np.zeros(len(thresholds)+2)
		tp[0] = 0; tp[-1] = 1
		fp[0] = 0; fp[-1] = 1
		for k, thresh in enumerate(thresholds):
			tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
			fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
		auc[rep] = np.trapz(tp, fp)
	return np.mean(auc) # Average across random splits

def NSS(saliency_map, fixation_map):
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)
	# Normalize saliency map to have zero mean and unit std
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	#cv2.imwrite('./Visualize/'+img[9:11]+'_NSS.png',normalize(s_map[f_map], method='range')*255)
	return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	cc = np.corrcoef(map1, map2)
	#cv2.imwrite('./Visualize/'+img[9:11]+'_CC.png',normalize(cc, method='range')*255)
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]


def SIM(saliency_map1, saliency_map2):
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='constant') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have values between [0,1] and sum up to 1
	map1 = normalize(map1, method='range')
	map2 = normalize(map2, method='range')
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')
	# Compute histogram intersection
	intersection = np.minimum(map1, map2)
	return np.sum(intersection)
#### METRICS --

# Name: func, symmetric?, second map should be saliency or fixation?
metrics = {
	"AUC_Judd": [AUC_Judd, False, 'fix'], # Binary fixation map
	"AUC_Borji": [AUC_Borji, False, 'fix'], # Binary fixation map
	"NSS": [NSS, False, 'fix'], # Binary fixation map
	"CC": [CC, False, 'sal'], # Saliency map
	"SIM": [SIM, False, 'sal'], # Saliency map
	"KLD": [KLD, False, 'sal'],
	"JSD": [JSD, False, 'sal'],
	 "KLD_Bernouli": [KLD_Bernouli, False, 'sal'],
	 "JSD_Bernouli": [JSD_Bernouli, False, 'sal']} # Saliency map

# Possible float precision of bin files
dtypes = {16: np.float16,
		  32: np.float32,
		  64: np.float64}

get_binsalmap_infoRE = re.compile("(\w+_\d{1,2})_(\d+)x(\d+)_(\d+)b")
def get_binsalmap_info(filename):
	name, width, height, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
	width, height, dtype = int(width), int(height), int(dtype)
	return name, width, height

def getSimVal(salmap1, salmap2, fixmap1=None, fixmap2=None):
	values = []

	for metric in keys_order:

		func = metrics[metric][0]
		sim = metrics[metric][1]
		compType = metrics[metric][2]

		if not sim:
			if compType == "fix" and not "NoneType" in [type(fixmap1), type(fixmap2)]:
				m = (func(salmap1, fixmap2)\
				   + func(salmap2, fixmap1))/2
			else:
				m = (func(salmap1, salmap2)\
				   + func(salmap2, salmap1))/2
		else:
			m = func(salmap1, salmap2)
		values.append(m)
	return values

def uniformSphereSampling(N):
	gr = (1 + np.sqrt(5))/2
	ga = 2 * np.pi * (1 - 1/gr)

	ix = iy = np.arange(N)

	lat = np.arccos(1 - 2*ix/(N-1))
	lon = iy * ga
	lon %= 2*np.pi

	return np.concatenate([lat[:, None], lon[:, None]], axis=1)

if __name__ == "__main__":
	from time import time
	t1 = time()
	# Similarité metrics to compute and output to file
	# keys_order = ['AUC_Judd', 'AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD']
	keys_order = ['AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD','JSD','KLD_Bernouli','JSD_Bernouli']

	# Head-only data
	# SM_PATH = "../H/SalMaps/"
	# SP_PATH = "../H/Scanpaths/"
	# Head-and-Eye data
	SM_PATH = "/home/yasser/Desktop/ftp.ivc.polytech.univ-nantes.fr/Images/HE/SalMaps/"
	SM_PATH1 = "./bin/"
	SP_PATH = "/home/yasser/Desktop/DATA360/Scanpaths/R"
	SP_PATH2 = "/home/yasser/Desktop/DATA360/Scanpaths/L"

	images = os.listdir(SM_PATH1)
	final = np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0])
	auc = []
	kld = []
	jsd = []
	cc = []
	sim = []
	nss = []
	fix = []
	i = 0
	import cv2


	#salmap3 = np.random.normal(loc=0,scale=1,size=(1024,2048))
	for img in images:
		SAMPLING_TYPE = [ # Different sampling method to apply to saliency maps
			"Sphere_9999999", # Too many points
			"Sphere_1256637", # 100,000 points per steradian
			"Sphere_10000",   # 10,000
			"Sin",			  # Sin(height)
			"Equi"			  # None
			]
		SAMPLING_TYPE = SAMPLING_TYPE[-2] # Sin weighting by default
		print("SAMPLING_TYPE: ", SAMPLING_TYPE)

		# Path to vieo saliency maps we wish to compare
		salmap1_path = SM_PATH +img
		salmap2_path = SM_PATH1 +img
		print(SP_PATH + "/HEscanpath_"+img[9:11]+'.txt')
		scanpath1_path = SP_PATH + "/HEscanpath_"+img[9:11]+'.txt'
		scanpath2_path = SP_PATH2 + "/HEscanpath_"+img[9:11]+'.txt'


		name1, width, height, = get_binsalmap_info(salmap1_path)
		name2, _, _, = get_binsalmap_info(salmap2_path)

		if SAMPLING_TYPE.split("_")[0] == "Sphere":
			print(int(SAMPLING_TYPE.split("_")[1]))
			unifS = uniformSphereSampling( int(SAMPLING_TYPE.split("_")[1]))
			unifS[:, 0] = unifS[:, 0] / np.pi * (height-1)
			unifS[:, 1] = unifS[:, 1] / (2*np.pi) * (width-1)
			unifS = unifS.astype(int)
		elif SAMPLING_TYPE == "Sin":
			VerticalWeighting = np.sin(np.linspace(0, np.pi, height)) # latitude weighting
			# plt.plot(np.arange(height), VerticalWeighting);plt.show()

		salmap1_file = open(salmap1_path, "rb")
		salmap2_file = open(salmap2_path, "rb")
		# Load from raw data
		salmap1 = np.fromfile(salmap1_file, count=width*height, dtype=np.float32)
		salmap2 = np.fromfile(salmap2_file, count=width*height, dtype=np.float32)
		salmap1 = salmap1.reshape([height, width])
		salmap2 = salmap2.reshape([height, width])
		# LOAD SM TO np.array
		salmap1_file.close()
		salmap2_file.close()

		# Load fixation lists
		fixations1 = np.loadtxt(scanpath1_path, delimiter=",", skiprows=1, usecols=(1,2))
		fixations2 = np.loadtxt(scanpath2_path, delimiter=",", skiprows=1, usecols=(1,2))
		fixations1 = fixations1 * [width, height] - [1,1]; fixations1 = fixations1.astype(int)
		fixations2 = fixations2 * [width, height] - [1,1]; fixations2 = fixations2.astype(int)

		# Create fixations maps
		fixmap1 = np.zeros(salmap1.shape, dtype=int)
		for iFix in range(fixations1.shape[0]):
			fixmap1[ fixations1[iFix, 1], fixations1[iFix, 0] ] += 1
		fixmap2 = np.zeros(salmap2.shape, dtype=int)
		for iFix in range(fixations2.shape[0]):
			fixmap2[ fixations2[iFix, 1], fixations2[iFix, 0] ] += 1

		# Apply uniform sphere sampling if specified
		if SAMPLING_TYPE.split("_")[0] == "Sphere":
			salmap1 = salmap1[unifS[:, 0], unifS[:, 1]]
			salmap2 = salmap2[unifS[:, 0], unifS[:, 1]]

			fixmap1 = fixmap1[unifS[:, 0], unifS[:, 1]]
			fixmap2 = fixmap2[unifS[:, 0], unifS[:, 1]]
		# Weight saliency maps vertically if specified
		elif SAMPLING_TYPE == "Sin":
			salmap1 = salmap1 * VerticalWeighting[:, None] + EPSILON
			salmap2 = salmap2 * VerticalWeighting[:, None] + EPSILON
		
		#f = fixmap1+fixmap2

		#fix.append(f.sum())

		
		
		#salmap2 = np.load('./priors/sitzman_prior.npy')
		#salmap2 = resize(salmap2,(1024,2048))
		salmap1 = normalize(salmap1, method='range')
		salmap2 = normalize(salmap2, method='range')

		# Compute similarity metricss
		values = getSimVal(salmap2, salmap1,
						   fixmap1,fixmap2)
		# Outputs results

		#visualize(salmap1,salmap2)
		i = i+1
		final = final + np.array(values)
		auc.append(values[0])
		cc.append(values[2])
		nss.append(values[1])
		sim.append(values[3])
		kld.append(values[4])
		jsd.append(values[5])
		print("stimName, metric, value")
		for iVal, val in enumerate(values):
			print("{}, {}, {}".format(name1, keys_order[iVal], val))

	print("T_delta = {}".format(time() - t1))
	
	print("******* The stds are *******")
	auc = np.array(auc)
	kld = np.array(kld)
	nss = np.array(nss)
	cc = np.array(cc)
	sim = np.array(sim)
	jsd = np.array(jsd)

	print('The std for AUC is: '+str(np.std(auc)))
	print('The std for NSS is: '+str(np.std(nss)))
	print('The std for CC is: '+str(np.std(cc)))
	print('The std for SIM is: '+str(np.std(sim)))
	print('The std for KLD is: '+str(np.std(kld)))
	print('The std for JSD is: '+str(np.std(jsd)))

	"""
	print("******* The max and min are *******")
	print('For AUC: The max value is: '+str(auc.max())+' The min value is '+str(auc.min()))
	print('For NSS: The max value is: '+str(nss.max())+' The min value is '+str(nss.min()))
	print('For CC: The max value is: '+str(cc.max())+' The min value is '+str(cc.min()))
	print('For SIM: The max value is: '+str(sim.max())+' The min value is '+str(sim.min()))
	print('For KLD: The max value is: '+str(kld.max())+' The min value is '+str(kld.min()))
	print("******* final *******")
	print(str(i))
	"""

	for iVal, val in enumerate(final / i):
		print("{}, {}, {}".format('final', keys_order[iVal], val))
