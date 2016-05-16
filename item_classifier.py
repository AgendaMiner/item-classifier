import pandas as pd
import numpy as np
import json
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics, svm, grid_search, ensemble


# to avoid truncating stuff
pd.set_option('display.max_colwidth', 800)
pd.set_option('display.max_rows', 1000)


def main():

	# features_df, info_df = prepDataset('data/training_data.csv')
	# datasets = splitTrainingTestingDFs(features_df, 'priority_wpusa_any', 'match_id')

	# write to disk to avoid expensive pre-processing every time
	# pickle.dump({'info_df':info_df, 'datasets':datasets}, open("data/processed_data.p", "wb" ))
	processed_data = pickle.load(open("data/processed_data.p", "rb" ))
	datasets = processed_data['datasets']
	info_df = processed_data['info_df']

	# classifyLasso(datasets)
	# classifyLogisticRegression(datasets, info_df)
	# classifySVM(datasets, info_df)
	classifyRandomForest(datasets, info_df)





'''
prepDataset
===========
Given the path to a csv file with training observations,
extract features from the CSV to build a df suitable for machine learning.
'''
def prepDataset(csv_filepath):
	info_df = pd.read_csv(csv_filepath)
	# info_df['match_id'] = info_df.index
	# print info_df['match_id'].head()

	# recode priority wpusa column as dummy vars
	info_df['priority_wpusa_high'] = np.where(info_df['priority_wpusa']=='yellow', 1, 0)
	info_df['priority_wpusa_low'] = np.where(info_df['priority_wpusa']=='blue', 1, 0)
	info_df['priority_wpusa_any'] = info_df['priority_wpusa_high'] + info_df['priority_wpusa_low']

	# code agency type
	with open('../agenda-parser/agencies_list.json') as data_file:
		agencies_list = json.load(data_file)
	agency_types = {agency['agency_id']: agency['agency_type'] for agency in agencies_list}
	info_df['agency_type'] = info_df['agency'].map(agency_types)

	# remove any lines where item text is NaN
	info_df = info_df.dropna(subset = ['item_text'])
	# redefine the index to fix merge issues with the dtm
	info_df.index = range(0,len(info_df))


	# convert text to DTM
	dtm = buildDTM(info_df, 'item_text')

	# remove unwanted columns - TRY MAKING SOME OF THESE INTO FEATURES LATER
	cols_to_drop = ['priority_sblc', 'priority_ibew', 'priority_unite', 'meeting_date', 'item_text_raw', 'meeting_sections.section_name', 'meeting_sections.section_number', 'item_type', 'item_recommendation', 'item_number', 'item_details','boarddocs_id', 'agency', 'priority_wpusa', 'item_text', 'priority_wpusa_high', 'priority_wpusa_low', 'agency_type']
	features_df = info_df.drop(cols_to_drop, axis=1)

	# merge with the DTM
	features_df = pd.concat([features_df, dtm], axis=1)

	# set the match id to be the index of the info df
	info_df.set_index('match_id', inplace=True)

	return [features_df, info_df]



'''
buildDTM
========
Given a dataframe and a column name, builds a document-term matrix
from the text of that column.
Returns the DTM as a pandas DF.
'''
def buildDTM(df, colname):
	text_list = df[colname].tolist()

	vectorizer = CountVectorizer(strip_accents="ascii", ngram_range=(1,3), stop_words='english', max_df=0.9, min_df=0.001) #### TODO - CUT DOWN THE NUMBER OF FEATURES
	counts_matrix = vectorizer.fit_transform(text_list)
	
	# convert counts matrix to pandas df
	return pd.DataFrame(counts_matrix.toarray(), columns=vectorizer.get_feature_names())



'''
splitTrainingTestingDFs
==============
Given a dataset, the name of a column containing the outcome variable, and the name of an ID column,
split the data into a training and testing set (each containing a matrix of X features and an array of Y indicators).
Return the split datasets as a dict.
'''
def splitTrainingTestingDFs(df, y_colname, id_colname):

	# split out the positive observations to ensure there occur in both training and testing data.
	pos_X_train, pos_X_test, pos_y_train, pos_y_test, pos_ids_train, pos_ids_test = splitObs(df, y_colname, id_colname, 1)
	neg_X_train, neg_X_test, neg_y_train, neg_y_test, neg_ids_train, neg_ids_test = splitObs(df, y_colname, id_colname, 0)

	# combine them back together
	X_train = np.vstack((pos_X_train, neg_X_train))
	X_test = np.vstack((pos_X_test, neg_X_test))
	y_train = np.concatenate((pos_y_train, neg_y_train), axis=0)
	y_test = np.concatenate((pos_y_test, neg_y_test), axis=0)
	ids_train = np.concatenate((pos_ids_train, neg_ids_train), axis=0)
	ids_test = np.concatenate((pos_ids_test, neg_ids_test), axis=0)

	# build a list of feature names
	df.drop([y_colname, id_colname], axis=1, inplace=True)
	feature_names = df.columns.values

	return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'ids_train': ids_train, 'ids_test': ids_test, 'feature_names': feature_names}



'''
splitObs
========
Given a dataframe, a y column name, an id column name, and a value of the y column,
splits the dataset into just the observations with that y value, and then
randomly into training and testing sets.
Returns 6 numpy arrays: X_train, X_test, y_train, y_test, ids_train, ids_test
'''
def splitObs(df, y_colname, id_colname, y_value):
	df = df[df[y_colname] == y_value]
	y = np.array(df[y_colname])
	ids = np.array(df[id_colname])
	features = np.array(df.drop([y_colname, id_colname], axis=1))
	return train_test_split(features, y, ids, test_size=0.33)



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



'''
classifyLogisticRegression
==========================
Builds a Logistic Regression classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyLogisticRegression(datasets, info_df):
	log_cv = linear_model.LogisticRegressionCV(cv=10, penalty='l1', scoring='recall', solver='liblinear', n_jobs=-1)
	log_cv.fit(datasets['X_train'], datasets['y_train'])
	preds = log_cv.predict(datasets['X_test'])
	evaluateModel(log_cv, datasets, preds)



'''
classifySVM
==========================
Builds an SVM classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifySVM(datasets, info_df):
	raw_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced')
	# search_params = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma': [0.0001, 0.01, 1, 10, 100]}
	search_params = {'kernel':['rbf'], 'C':[1, 10], 'gamma': [0.0001]}
	svm_cv = grid_search.GridSearchCV(raw_svm, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5)
	svm_cv.fit(datasets['X_train'], datasets['y_train'])
	print(svm_cv.best_params_)
	preds = svm_cv.predict(datasets['X_test'])
	evaluateModel(svm_cv, datasets, preds)



'''
classifyRandomForest
==========================
Builds a random forest classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyRandomForest(datasets, info_df):
	rand_forest = ensemble.RandomForestClassifier(n_estimators=20, oob_score=True, n_jobs=-1, verbose=5, class_weight='balanced')
	search_params = {'max_depth':[10,100,1000]}
	rand_forest_cv = grid_search.GridSearchCV(rand_forest, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5)

	rand_forest_cv.fit(datasets['X_train'], datasets['y_train'])
	preds = rand_forest_cv.predict(datasets['X_test'])
	evaluateModel(rand_forest_cv, datasets, preds)
	# printNonZeroFeatures(rand_forest_cv, datasets)



'''
evaluateModel
=============
Prints out a variety of evaluation metrics to see how
well the model performs.
'''
def evaluateModel(model, datasets, preds):
	# printNonZeroCoefficients(log_cv, datasets)
	print(metrics.classification_report(datasets['y_test'], preds))
	print(metrics.confusion_matrix(datasets['y_test'], preds))
	# matchPredsToInfoDF(datasets['y_test'], preds, datasets['ids_test'], info_df)


'''
printNonZeroCoefficients
========================
Print out the coefficients from regularized model that haven't been set to zero,
to help with evaluating models.
'''
def printNonZeroCoefficients(model, datasets):
	coefs = pd.DataFrame({'feature':datasets['feature_names'], 'coefficient':model.coef_[0]})
	print(coefs[coefs['coefficient'] > 0])




'''
printNonZeroFeatures
========================
Print out the features from a random forest model that haven't been set to zero,
to help with evaluating models.
'''
def printNonZeroFeatures(model, datasets):
	coefs = pd.DataFrame({'feature':datasets['feature_names'], 'coefficient':model.feature_importances_})
	print(coefs[coefs['coefficient'] > 0])



'''
matchPredsToInfoDF
==================
Uses the match id to match predictions up with the original agenda items
to get a better sense of what the model is catching/missing.
'''
def matchPredsToInfoDF(true_y, pred_y, ids, info_df):
	# build df of preds, with id as index
	preds_df = pd.DataFrame({'true_y': true_y, 'pred_y': pred_y}, index=ids)
	combined_df = info_df.join(preds_df, how='inner')

	print("============MATCHED SUCCESSFULLY===========")
	print(combined_df.loc[(combined_df['true_y'] == 1) & (combined_df['pred_y']==1), 'item_text'])

	print("============FALSE NEGATIVE===========")
	print(combined_df.loc[(combined_df['true_y'] == 1) & (combined_df['pred_y']==0), 'item_text'])

	print("============FALSE POSITIVE===========")
	print(combined_df.loc[(combined_df['true_y'] == 0) & (combined_df['pred_y']==1), 'item_text'])






if __name__ == '__main__':
    main()