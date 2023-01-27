import csv
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from time import perf_counter
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import losses
from urllib.parse import urlparse

#Code from https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
def split_dataset(dataset, test_ratio=0.30):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]


#contains the URLs from the dataset
datasetURLs = []
#contains the value determining if a website is Valid(1) or Invalid(0)
phishingValidity = []

#where the datasets information is parsed into distinct features
simpleDict = {'full_url': [],
	'url_special_characters': [], 
	'url_length': [], 
	'num_to_char_ratio': [], 
	"is_phishing": []}
	
complexDict = {'full_url': [],
	'protocol': [],
	'domain': [], 
	'path': [], 
	'query': [], 
	'fragment': [],  
	'url_special_characters': [], 
	'url_length': [], 
	'num_to_char_ratio': [], 
	"is_phishing": []}

#counter values for the while loops below
x = 0
y = 0

#imports the dataset used and writes the information to workable format
with open('CatchPhish.csv', newline='') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		datasetURLs.append(row['Url'])
		phishingValidity.append(row['class'])

#populates simpleDict with values from the dataset imported
while x < len(datasetURLs):
	numChar = 0
	numSpecial = 0
	numNumber = 0
	for character in datasetURLs[x]:
		if(character.isdigit()):
			numNumber += 1
		elif(character.isalpha()):
			numChar += 1
		else:
			numSpecial += 1
	simpleDict['full_url'].append(datasetURLs[x])
	simpleDict['url_special_characters'].append(numSpecial)
	simpleDict['url_length'].append(len(datasetURLs[x]))
	simpleDict['num_to_char_ratio'].append(numChar/(len(datasetURLs[x])))
	simpleDict['is_phishing'].append(phishingValidity[x])
	
	x += 1

#populates complexDict with values from the dataset imported
while y < len(datasetURLs):
	numChar = 0
	numSpecial = 0
	numNumber = 0
	parsedURL = urlparse(datasetURLs[y])
	for character in datasetURLs[y]:
		if(character.isdigit()):
			numNumber += 1
		elif(character.isalpha()):
			numChar += 1
		else:
			numSpecial += 1
	complexDict['full_url'].append(datasetURLs[y])
	complexDict['url_special_characters'].append(numSpecial)
	complexDict['url_length'].append(len(datasetURLs[y]))
	complexDict['num_to_char_ratio'].append(numChar/(len(datasetURLs[y])))
	complexDict['is_phishing'].append(phishingValidity[y])
	complexDict['protocol'].append(parsedURL.scheme)
	complexDict['domain'].append(parsedURL.netloc)
	complexDict['path'].append(parsedURL.path)
	complexDict['query'].append(parsedURL.query)
	complexDict['fragment'].append(parsedURL.fragment)
	
	y += 1

#Code based on https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
#Creates Pandas dataframes of the dataset, and then converts into Testing and Training Keras sets divided into simple (5 features) and complex (10 features)
simple_dataframe = pd.DataFrame(simpleDict)
complex_dataframe = pd.DataFrame(complexDict)
train_simple_ds_pd, test_simple_ds_pd = split_dataset(simple_dataframe)
train_complex_ds_pd, test_complex_ds_pd = split_dataset(complex_dataframe)
simple_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_simple_ds_pd, label='is_phishing')
simple_test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_simple_ds_pd, label='is_phishing')
complex_train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_complex_ds_pd, label='is_phishing')
complex_test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_complex_ds_pd, label='is_phishing')

'''
print("/n" + "<== Simple Evals ==>")
srf_time1 = perf_counter()
simple_rf_phish_model = tfdf.keras.RandomForestModel(verbose=2)
simple_rf_phish_model.fit(x=simple_train_ds)
srf_time2 = perf_counter()
#simple_rf_phish_model.save("/modelMaker/simple_rf_model")

sgbt_time1 = perf_counter()
simple_gbt_phish_model = tfdf.keras.GradientBoostedTreesModel(verbose=2)
simple_gbt_phish_model.fit(x=simple_train_ds)
sgbt_time2 = perf_counter()
#simple_gbt_phish_model.save("/modelMaker/simple_gbt_model")
'''
#Code that evalutes a Random Forest Model with a simple URL feature set and tracks the time taken to create the model
'''
evaluation = simple_rf_phish_model.evaluate(simple_test_ds, return_dict=True)
print()
for name, value in evaluation.items():
	print(f"{name}: {value:.4f}")
'''

#Code that evalutes a Gradient Boosted Tree with a simple URL feature set and tracks the time taken to create the model
'''
evaluation = simple_gbt_phish_model.evaluate(simple_test_ds, return_dict=True)
print()
for name, value in evaluation.items():
	print(f"{name}: {value:.4f}")
'''


print("/n" + "<== Complex Evals ==>")

#Code that trains a Random Forest Model with a complex URL feature set and tracks the time taken to create the model
crf_time1 = perf_counter()
complex_rf_phish_model = tfdf.keras.GradientBoostedTreesModel(verbose=2)
complex_rf_phish_model.fit(x=complex_train_ds)
crf_time2 = perf_counter()
complex_rf_phish_model.save("modelMaker/complex_rf_model")

#Code that Trains a Gradient Boosted Tree Model with a complex URL feature set and tracks the time taken to create the model
cgbt_time1 = perf_counter()
complex_gbt_phish_model = tfdf.keras.GradientBoostedTreesModel(verbose=2)
complex_gbt_phish_model.fit(x=complex_train_ds)
cgbt_time2 = perf_counter()
complex_gbt_phish_model.save("modelMaker/complex_gbt_model")

#Code to track time to create model
'''
print("Time to create models")
print(srf_time2 - srf_time1)
print(sgbt_time2 - sgbt_time1)
print(crf_time2 - crf_time1)
print(cgbt_time2 - cgbt_time1)
'''
#Code to evaluate a Random Forest Model with a complex URL feature set
'''
evaluation = complex_rf_phish_model.evaluate(complex_test_ds, return_dict=True)
print()
for name, value in evaluation.items():
	print(f"{name}: {value:.4f}")
'''

#Code to evaluate a Gradient Boosted Tree Model with a complex URL feature set
'''
evaluation = complex_gbt_phish_model.evaluate(complex_test_ds, return_dict=True)
print()
for name, value in evaluation.items():
	print(f"{name}: {value:.4f}")
'''
