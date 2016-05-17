import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model, metrics, svm, grid_search, ensemble, neighbors, naive_bayes


# to avoid truncating stuff
pd.set_option('display.max_colwidth', 800)
pd.set_option('display.max_rows', 1000)


def main():

	datasets = prepDatasets('data/training_data.csv')

	# classifyLasso(datasets)
	# classifyLogisticRegression(datasets, 'priority_sblc_wpusa', 'match_id')
	classifySVM(datasets, 'priority_sblc_wpusa', 'match_id')
	classifyNaiveBayes(datasets, 'priority_sblc_wpusa', 'match_id')
	# classifyRandomForest(datasets, info_df)
	# classifyNearestNeighbors(datasets, 'priority_sblc_wpusa', 'match_id')





'''
prepDatasets
============
Given the path to a csv file with training observations,
extract features from the CSV to build a df suitable for machine learning.
'''
def prepDatasets(csv_filepath):
	info_df = pd.read_csv(csv_filepath)

	# recode priority columns as dummy vars
	info_df['priority_wpusa_high'] = np.where(info_df['priority_wpusa']=='yellow', 1, 0)
	info_df['priority_wpusa_low'] = np.where(info_df['priority_wpusa']=='blue', 1, 0)
	info_df['priority_wpusa_any'] = info_df['priority_wpusa_high'] + info_df['priority_wpusa_low']

	info_df['priority_sblc_high'] = np.where(info_df['priority_sblc']=='yellow', 1, 0)
	info_df['priority_sblc_low'] = np.where(info_df['priority_sblc']=='blue', 1, 0)
	info_df['priority_sblc_any'] = info_df['priority_sblc_high'] + info_df['priority_sblc_low']

	info_df['priority_sblc_wpusa'] = np.where((info_df['priority_wpusa_any']==1) | (info_df['priority_sblc_any']==1), 1, 0)

	# code agency type
	with open('../agenda-parser/agencies_list.json') as data_file:
		agencies_list = json.load(data_file)
	agency_types = {agency['agency_id']: agency['agency_type'] for agency in agencies_list}
	info_df['agency_type'] = info_df['agency'].map(agency_types)

	# remove any lines where item text is NaN
	info_df = info_df.dropna(subset = ['item_text'])
	# redefine the index to fix merge issues with the dtm
	info_df.index = range(0,len(info_df))

	# # remove unwanted columns - TRY MAKING SOME OF THESE INTO FEATURES LATER
	# cols_to_drop = ['priority_sblc', 'priority_ibew', 'priority_unite', 'meeting_date', 'item_text_raw', 'meeting_sections.section_name', 'meeting_sections.section_number', 'item_type', 'item_recommendation', 'item_number', 'item_details','boarddocs_id', 'agency', 'priority_wpusa', 'item_text', 'priority_wpusa_high', 'priority_wpusa_low', 'agency_type']
	# features_df = info_df.drop(cols_to_drop, axis=1)

	# # merge with the DTM
	# features_df = pd.concat([features_df, dtm], axis=1)

	# set the match id to be the index of the info df
	info_df.set_index('match_id', inplace=True, drop=False)

	datasets = {'info_df': info_df}
	dtm_parts = buildDTM(info_df, 'item_text')
	datasets['dtm'] = dtm_parts[0]
	datasets['dtm_terms'] = dtm_parts[1]

	return datasets



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
	
	return [counts_matrix, vectorizer.get_feature_names()]



'''
splitObs
========
Given a matrix of features, a dataframe, a y column name, and an id column name,
splits the dataset randomly into training and testing sets.
Returns 6 arrays: X_train, X_test, y_train, y_test, ids_train, ids_test
'''
def splitObs(features, df, y_colname, id_colname):
	y = np.array(df[y_colname])
	ids = np.array(df[id_colname])
	return train_test_split(features, y, ids, test_size=0.33, stratify=y)



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
def classifyLogisticRegression(datasets, y_colname, id_colname):

	X_train, X_test, y_train, y_test, ids_train, ids_test = splitObs(datasets['dtm'], datasets['info_df'], y_colname, id_colname)

	log_cv = linear_model.LogisticRegressionCV(cv=10, penalty='l1', scoring='recall', solver='liblinear', n_jobs=-1)
	log_cv.fit(X_train, y_train)
	preds = log_cv.predict(X_test)

	# evaluateModel(log_cv, datasets, preds)

	print(metrics.classification_report(y_test, preds))
	print(metrics.confusion_matrix(y_test, preds))



'''
classifySVM
==========================
Builds an SVM classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifySVM(datasets, y_colname, id_colname):

	X_train, X_test, y_train, y_test, ids_train, ids_test = splitObs(datasets['dtm'], datasets['info_df'], y_colname, id_colname)

	# cross validate to find best hyperparameters
	# raw_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced')
	# search_params = {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma': [0.00001, 0.0001, 0.01, 1, 10, 100]}
	# # search_params = {'kernel':['rbf'], 'C':[1, 10], 'gamma': [0.0001]}
	# svm_cv = grid_search.GridSearchCV(raw_svm, param_grid=search_params, scoring='recall', cv=5, n_jobs=-1, verbose=5)
	# svm_cv.fit(X_train, y_train)
	# print(svm_cv.best_params_)

	# use best hyperparameters in the future (uncomment above and comment out this line to re-cross validate)
	best_svm = svm.SVC(decision_function_shape='ovr', class_weight='balanced', kernel='rbf', C=100, gamma=0.0001)
	best_svm.fit(X_train, y_train)

	preds = best_svm.predict(X_test)

	print(metrics.classification_report(y_test, preds))
	print(metrics.confusion_matrix(y_test, preds))
	# matchPredsToInfoDF(y_test, preds, ids_test, datasets['info_df'])


'''
classifyNearestNeighbors
==========================
Builds a nearest neighbors classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyNearestNeighbors(datasets, y_colname, id_colname):

	X_train, X_test, y_train, y_test, ids_train, ids_test = splitObs(datasets['dtm'], datasets['info_df'], y_colname, id_colname)

	knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1)
	knn.fit(X_train, y_train)
	preds = knn.predict(X_test)

	print(metrics.classification_report(y_test, preds))
	print(metrics.confusion_matrix(y_test, preds))




'''
classifyNaiveBayes
==========================
Builds a naive Bayes classifier from the training dataset,
then tests how well it performs on the testing data.
'''
def classifyNaiveBayes(datasets, y_colname, id_colname):

	X_train, X_test, y_train, y_test, ids_train, ids_test = splitObs(datasets['dtm'], datasets['info_df'], y_colname, id_colname)

	nb_model = naive_bayes.MultinomialNB(alpha=0.5)
	nb_model.fit(X_train, y_train)
	preds = nb_model.predict(X_test)

	print(metrics.classification_report(y_test, preds))
	print(metrics.confusion_matrix(y_test, preds))




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