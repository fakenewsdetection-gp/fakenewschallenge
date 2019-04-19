import pandas as pd
import pickle
from preprocess import preprocess_data
from features.TfidfFeatureGenerator import TfidfFeatureGenerator


# Flags
read_flag = True
test_flag = True
generators = [
	TfidfFeatureGenerator()
]


def process():
	if read_flag:
		# Read training set
		body_train = pd.read_csv('fnc-1/train_bodies.csv', encoding='utf-8')
		stances_train = pd.read_csv('fnc-1/train_stances.csv', encoding='utf-8')

		train = pd.merge(stances_train, body_train, how='left', on='Body ID')
		targets = ['agree', 'disagree', 'discuss', 'unrelated']
		targets_dict = dict(zip(targets, range(len(targets))))
		train['target'] = map(lambda x: targets_dict[x], train['Stance'])
		
		data = train

		# Read testing set, concatenate both training and testing set in data.
		if test_flag:
			body_test = pd.read_csv('fnc-1/test_bodies.csv', encoding='utf-8')
			headline_test = pd.read_csv('fnc-1/test_stances_unlabeled.csv', encoding='utf-8')
			test = pd.merge(headline_test, body_test, how='left', on='Body ID')

			data = pd.concat((train, test))
			
			train = data[~data['target'].isnull()]
			test = data[data['target'].isnull()]

			print('Training set shape: ', train.shape)
			print('Testing set shape: ', test.shape)

		# Generate unigrams
		print('Generating unigrams...')
		data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))
		data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))

		# Write to file
		with open('data.pkl', 'wb') as outfile:
			pickle.dump(data, outfile, -1)
			print('Dataframe written in data.pkl')
	else:
		with open('data.pkl', 'rb') as infile:
			data = pickle.load(infile)
			print("Loaded data from data.pkl")
			print("Data shape: ", data.shape)

	

if __name__ == '__main__':
	process()

