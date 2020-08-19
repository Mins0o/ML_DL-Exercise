import csv
import numpy as np
import matplotlib.pyplot as plt

def TsvToLine(filePath):
	with open(filePath,'r') as file:
		lines=list(csv.reader(file,delimiter='\t'))
		data=[]
		for line in lines:
			try:
				singleLine=([[int(x) for x in col.split(',')] for col in line[:-1]],line[-1])
			except ValueError:
				singleLine=([[x for x in col.split(',')] for col in line[:-1]],line[-1])
			data.append(singleLine)
	return data
	
def TsvToXY(filePath):
	with open(filePath,'r') as file:
		lines=list(csv.reader(file,delimiter='\t'))
		data=[]
		for line in lines:
			listedLine=[]
			for collumn in line[:-1]:
				try:
					if len(line[:-1])==1:
						listedLine=[int(x) for x in collumn.split(',')]
					else:
						listedLine.append([int(x) for x in collumn.split(',')])
				except ValueError:
					if len(line[:-1])==1:
						listedLine=[int(x) for x in collumn.split(',')]
					else:
						listedLine.append([int(x) for x in collumn.split(',')])
			data.append(listedLine)
		label=[line[-1] for line in lines]
	return(data,label)
	
def FourierTransform(signal,samplingRate,verbose=False):
	timeAxis=np.arange(0,len(signal)/samplingRate,1/samplingRate,dtype="double")
	frequencyCap=samplingRate//2
	if not type(signal) == np.ndarray:
		try:
			signal=np.array(signal)
		except:
			print("FourierTransform(): Please input an numpy array")
			return []
			
	signal=normalize(signal)
	
	frequencyChart=createFrequencyChart(frequencyCap)
	"""
	timeAxis = [t1,t2,...,tn]
	frequencyChart = [f1,f2,...,fm]
	
	What I need is:
	
							(> time axis >)					(> time axis >)						(> time axis >)
	(v freq axis v) sum of ( signal[t1] * exp(j*2pi*t1*f1), signal[t2] * exp(j*2pi*t2*f1), ... , signal[tn] * exp(j*2pi*tn*f1) )
	(v freq axis v) sum of ( signal[t1] * exp(j*2pi*t1*f2), signal[t2] * exp(j*2pi*t2*f2), ... , signal[tn] * exp(j*2pi*tn*f2) )
	...														...										...
	...														...										...
	...														...										...
	(v freq axis v) sum of ( signal[t1] * exp(j*2pi*t1*fm), signal[t2] * exp(j*2pi*t2*fm), ... , signal[tn] * exp(j*2pi*tn*fm) )
	
	so, (summed and transposed from above)
		(> freq axis >)		(> freq axis >)
	fourier = ( sum_1, sum_2, ... , sum_m)
	
	this line,
	>>> np.stack([timeAxis for i in range(len(frequencyChart))],axis=1)
	makes
	| t1 t1 t1 ... t1 |
	| t2 t2 t2 ... t2 |
	|       ...       |
	| tn tn tn ... tn |
	
	multiplying frequencyChart
	>>> | f1 f2 ... fm | * prevArray
	makes
	| t1*f1 t1*f2 t1*f3 ... t1*fm |
	| t2*f1 t2*f2 t2*f3 ... t2*fm |
	|       ...         ...       |
	| tn*f1 tn*f2 tn*f3 ... tn*fm |
	
	now, multiply 2*pi*(0+1j) and put it in exponential
	then dot product signal to get fourier transform for each frequencies
	
	"""
	f_tPlane=np.stack([timeAxis for i in range(len(frequencyChart))],axis=1)*frequencyChart
	imExpPlane=np.exp(f_tPlane*2*np.pi*(0+1j))
	fourier=signal.dot(imExpPlane)
	if verbose:
		plt.subplot(2,1,1)
		plt.plot(timeAxis,signal)
		plt.subplot(2,1,2)
		plt.plot(frequencyChart,np.abs(fourier))
		plt.show()
	return fourier
	
			
	
def normalize(signal):
	temp=signal-signal.min()
	temp=temp/temp.std()
	return(temp-temp.mean())
	
def createFrequencyChart(freqCap):
	# https://musicproductiontips.net/wp-content/uploads/pdf/musicproductiontips.net-Frequency_Chart__The_Most_Important_Audio_Frequency_Ranges-A4.pdf
	return np.array(
		list(range(20,min(60,freqCap),20))+list(range(60,min(120,freqCap),10))+list(range(120,min(350,freqCap),5))+list(range(350,min(2000,freqCap),20))+list(range(2000,min(8000,freqCap),100))+list(range(8000,min(20000,freqCap),300)))

if(__name__=="__main__"):
	temp_x,y=TsvToXY('D:/Dropbox/Workspace/03 Python/03 ML_DL_Correlation_Convolution-Exercise/Data/SoundPCM/PCM_45kHz.tsv')
	x=np.array([np.array(data) for data in temp_x])
	for i in range(len(x)):
		print(i, end="\t{0}\n\n".format(y[i]))
		FourierTransform(x[i],7840)