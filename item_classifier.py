import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics




def main():

	df = prepDataset('data/training_data.csv')
	datasets = splitTrainingTestingDFs(df, 'priority_wpusa_any')
	# classifyLasso(datasets)
	classifyLogisticRegression(datasets)





'''
prepDataset
===========
Given the path to a csv file with training observations,
extract features from the CSV to build a df suitable for machine learning.
'''
def prepDataset(csv_filepath):
	df = pd.read_csv(csv_filepath)

	# recode priority wpusa column as dummy vars
	df['priority_wpusa_high'] = np.where(df['priority_wpusa']=='yellow', 1, 0)
	df['priority_wpusa_low'] = np.where(df['priority_wpusa']=='blue', 1, 0)
	df['priority_wpusa_any'] = df['priority_wpusa_high'] + df['priority_wpusa_low']

	# code agency type
	with open('../agenda-parser/agencies_list.json') as data_file:
		agencies_list = json.load(data_file)
	agency_types = {agency['agency_id']: agency['agency_type'] for agency in agencies_list}
	df['agency_type'] = df['agency'].map(agency_types)

	# convert text to DTM
	dtm = buildDTM(df, 'item_text')

	# remove unwanted columns - TRY MAKING SOME OF THESE INTO FEATURES LATER
	cols_to_drop = ['match_id', 'priority_sblc', 'priority_ibew', 'priority_unite', 'meeting_date', 'item_text_raw', 'meeting_sections.section_name', 'meeting_sections.section_number', 'item_type', 'item_recommendation', 'item_number', 'item_details','boarddocs_id', 'agency', 'priority_wpusa', 'item_text', 'priority_wpusa_high', 'priority_wpusa_low', 'agency_type']
	df = df.drop(cols_to_drop, axis=1)

	# merge with the DTM
	df = pd.concat([df, dtm], axis=1)

	return df



'''
buildDTM
========
Given a dataframe and a column name, builds a document-term matrix
from the text of that column.
Returns the DTM as a pandas DF.
'''
def buildDTM(df, colname):
	text_list = df[colname].tolist()

	vectorizer = CountVectorizer(strip_accents="ascii", ngram_range=(1,3), min_df=0.01) #### TODO - CUT DOWN THE NUMBER OF FEATURES
	counts_matrix = vectorizer.fit_transform(text_list)

	# convert counts matrix to pandas df
	return pd.DataFrame(counts_matrix.toarray(), columns=vectorizer.get_feature_names())



'''
splitTrainingTestingDFs
==============
Given a dataset and the name of a column containing the outcome variable,
split the data into a training and testing set (each containing a matrix of X features and an array of Y indicators).
Return the split datasets as a dict.
'''
def splitTrainingTestingDFs(df, y_colname):
	y_series = df[y_colname]

	# split out the positive observations to ensure there occur in both training and testing data.
	pos_X_train, pos_X_test, pos_y_train, pos_y_test = splitObs(df, y_colname, 1)
	neg_X_train, neg_X_test, neg_y_train, neg_y_test = splitObs(df, y_colname, 0)

	# combine them back together
	X_train = np.vstack((pos_X_train, neg_X_train))
	X_test = np.vstack((pos_X_test, neg_X_test))
	y_train = np.concatenate((pos_y_train, neg_y_train), axis=0)
	y_test = np.concatenate((pos_y_test, neg_y_test), axis=0)
	print pos_X_train.shape
	print neg_X_train.shape
	print X_train.shape

	return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}



'''
splitObs
========
Given a dataframe, a y column indicator, and a value of the y column,
splits the dataset into just the observations with that y value, and then
randomly into training and testing sets.
Returns 4 numpy arrays: X_train, X_test, y_train, y_test
'''
def splitObs(df, y_colname, y_value):
	df = df[df[y_colname] == y_value]
	y = np.array(df[y_colname])
	features = np.array(df.drop(y_colname, axis=1))
	return train_test_split(features, y, test_size=0.33)



'''
classifyLasso
=============
Builds a LASSO classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyLasso(datasets):

	lasso_cv = linear_model.LassoCV(cv = 10, verbose = True)
	lasso_cv.fit(datasets['X_train'], datasets['y_train'])
	raw_preds = lasso_cv.predict(datasets['X_test'])
	pred_classes = np.where(raw_preds >= 0.5, 1, 0)
	print pred_classes
	print(metrics.accuracy_score(datasets['y_test'], pred_classes))
	# print(metrics.precision_recall_fscore_support(datasets['y_test'], pred_classes))


'''
classifyLogisticRegression
==========================
Builds a Logistic Regression classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyLogisticRegression(datasets):
	log_cv = linear_model.LogisticRegressionCV(cv=10, penalty='l1', scoring='recall', solver='liblinear', n_jobs=-1, verbose=1)
	log_cv.fit(datasets['X_train'], datasets['y_train'])
	preds = log_cv.predict(datasets['X_test'])
	print(metrics.classification_report(datasets['y_test'], preds))
	print(metrics.confusion_matrix(datasets['y_test'], preds))
	print(preds)










if __name__ == '__main__':
    main()