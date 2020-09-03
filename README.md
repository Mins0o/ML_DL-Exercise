# ML_DL_Correlation_Convolution-Exercise
Trying to understand the difference between ML and DL and compare Convolution and Cross Correlation.
### Motivation
On my recent project, [Door Opener](https://github.com/Mins0o/Door_Opener), I used my limited knowledge of machine learning to classify audio signal.  
I wanted to dive deeper into the study of machine learning with this project. After taking a one week crash course at my Univ about DL now I have a better understanding of how it works.  
This is my attempt to get familiar with the subject.
  
For these exercises, I will be doing supervised classifications.  
  
___
## Machine Learning
### Name-Gender Classification
This is how I first encountered the world of machine learning. I took an NLP course last semester and learned about classifiers for the first time. I thought revisiting this example will be perfect as a warm up.
- Data  
  For name data, I used the python nltk package's builtin names dataset (`nltk.corpus.names`).
- (data) Features  
  - Last letter: I happen to know the last letter of a name somehow works well as an indicater of gender. 
  - First two letters: This was a random choice I made.
  - Length of the name
  - Number of 'm's it contains
  - Number of double 'l's it contains
  - If the name ends with "ce"
- Classifiers  
  Since I didn't know the differences of the classifiers, I tried a bunch of classifiers provided from the sklearn package.
  The categorical NB classifier seem to work well.
  I played around with the features and made about 2% of consistant improvment with the categorial classifier.
- Result  
  75% ~ 78% accuracy. (regardless of train-test split)
   
Read more about how to do this experiment with nltk [here](https://www.geeksforgeeks.org/python-gender-identification-by-name-using-nltk/).

  
### Audio Classification
This is an extension from my last project, the [Door Opener](https://github.com/Mins0o/Door_Opener). Carrying on from the project, I wanted to experience machine learning development process in more extent: Data acquiring, preprocessing, feature extraction and classifier experiments.
- Data  
  For this experiment, I used [custom made datasets](https://github.com/Mins0o/PCMLabeler "PCMLabeler Repository"). PCM data of air pressure sampled in 45000 Hz and 7840 Hz.  
  The dataset has about 130 instances with 4 categories of sound waves: tap on mic, finger snap, clap, and whistling.
- Preprocessing
  I was experimenting with the recording settings in the recording project, so the data had different length, sampling rate etc. I trimed/elongated/resampled/interpolated the data to match recordings from several different settings.
- (data) Features  
  - Most dominant frequency: Fourier transform was used for frequency spectrum analysis.  
  - Duration: Since whistling had significantly longer duration than other sounds which are basically pulses, I add a feature to measure duration of significant air pressure change.
  - Mid-range most dominant frequency: The pulses were having occasional low-frequency spikes, so I added this as another feature to differentiate them.
- Classifiers
  Categorical, Gaussian, Tree classifiers, and SVC seemed to work well (not in particular order), but the categorical classifier had an [issue with unseen value at training](github.com/scikit-learn/scikit-learn/pull/16326). 
- Result  
  The accuracy depends on the random train-test split, but it ranges from 70% ~ 100% (trained on 80% of the dataset) at average of about 80%. Considering how small the dataset is, I think the result is pretty decent.
___
## Deep Learning
### Mnist Classification
### Audio Classification