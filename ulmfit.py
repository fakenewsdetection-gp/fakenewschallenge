import pandas as pd
import pickle
import argparse
from preprocess import preprocess_data
from sklearn.model_selection import train_test_split
# from fastai.text import *


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--no-read", type=bool, help="read flag",
					default=False, const=True, nargs='?')
	parser.add_argument("-n", type=int, help="number of instances",
					default=-1, required=False)
	parser.add_argument("--ulm-path", type=str, help="path to ulmfit data",
					default="./ulm", required=False)
	parser.add_argument("--load-ulm", type=bool, help="load ulmfit",
					default=False, const=True, nargs='?')
	return parser.parse_args()


def process(args):
	def _cat_headline_body(x):
		res = '%s <ENDTITLE>\n%s' % (x['Headline'], x['articleBody'])
		return res

	if not args.no_read:
		# Read training set
		body_train = pd.read_csv('fnc-1/train_bodies.csv', encoding='utf-8')
		stances_train = pd.read_csv('fnc-1/train_stances.csv', encoding='utf-8')

		train = pd.merge(stances_train, body_train, how='left', on='Body ID')
		targets = ['agree', 'disagree', 'discuss', 'unrelated']
		targets_dict = dict(zip(targets, range(len(targets))))
		train['target'] = train['Stance'].map(lambda x: targets_dict[x])

		if args.n != -1:
			train = train[:args.n]
		
		body_test = pd.read_csv('fnc-1/test_bodies.csv', encoding='utf-8')
		headline_test = pd.read_csv('fnc-1/test_stances_unlabeled.csv', encoding='utf-8')
		test = pd.merge(headline_test, body_test, how='left', on='Body ID')

		if args.n != -1:
			test = test[:args.n]

		data = pd.concat((train, test))
			
		# Write to file
		data.to_pickle('data.pkl')
		
	else:
		data = pd.read_pickle('data.pkl')
		print('Loaded data from data.pkl')
		print('Data shape: ', data.shape)

	data['all_text'] = list(data.apply(_cat_headline_body, axis=1))
	n_train = data[~data['target'].isnull()].shape[0]

	train = data.iloc[:n_train, :]
	train, val = train_test_split(train, test_size=0.2, random_state=7, stratify=train['target'])
	test = data.iloc[n_train:, :]

	return train, val, test


def train_model(train, val, test, args):
	if not args.load_uml:
		data_lm = TextLMDataBunch.from_df(args.ulm_path, train, val,
							text_cols=['all_text'], label_cols='target')
		data_cl = TextClasDataBunch.from_df(args.ulm_path, train, val,
							vocab=data_lm.train_ds.vocab, bs=32, text_cols=['all_text'], label_cols='target')
		data_lm.save('data_lm.pkl')
		data_cl.save('data_cl.pkl')
	else:
		data_lm = load_data(args.ulm_path, 'data_lm.pkl')
		data_cl = load_data(args.ulm_path, 'data_clas.pkl', bs=32)

	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
	learn.fit_one_cycle(1, 1e-2)



if __name__ == '__main__':
	args = parse_args()
	print(args)
	train, val, test = process(args)
	train_model(train, val, test, args)


