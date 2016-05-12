import pandas as pd
import os
import json
import pickle
import re
from nltk.tokenize import sent_tokenize

pd.set_option('display.max_colwidth', 800)

def main():

	with open('../agenda-parser/agencies_list.json') as data_file:
		parsed_agencies = json.load(data_file)

	reports_df = buildDataFrameOfReports('../agenda-parser/docs/training_data/structured_reports/')
	reports_df.to_csv("data/all_report_items.csv", encoding="utf-8", index=False)

	items_df = buildDataFrameOfAgendaItems('../agenda-parser/docs', parsed_agencies)

	matchReportsToItems(reports_df, items_df, parsed_agencies)
	writeTrainingDataToDisk(items_df)



'''
buildDataFrameOfReports
=======================
Given the path to a directory containing JSON-structured
lists of agenda items that have been hand-coded as interesting,
converts the JSON files to Pandas DFs, then concat all the DFs together.
Returns the combined DF.
'''
def buildDataFrameOfReports(directory_path):

	df_list = list()
	for filename in os.listdir(directory_path):
		if filename.endswith(".json"):
			filepath = os.path.join(directory_path, filename)
			df = pd.read_json(filepath)
			df['report_name'] = filename
			df_list.append(df)

	return pd.concat(df_list, ignore_index=True)



'''
buildDataFrameOfAgendaItems
===========================
Given a list of parsed agencies and the path to a directory containing the data for each parsed agency, 
get all the structured agendas, convert them to a Pandas DF, then concat all the DFs together.
If there is a csv file containing hand-matched items, load this as well, subset it to hand-matched items,
and join this with the other DF.
Returns the combines DF.
'''
def buildDataFrameOfAgendaItems(base_dir, parsed_agencies):

	# get list of agencies with parsed agendas
	agencies_list = list()
	for dir in os.listdir(base_dir):
		if dir in [agency['agency_id'] for agency in parsed_agencies]:
			agencies_list.append(dir)

	df_list = list()

	# get all the agenda files from each agency
	for agency in agencies_list:
		agendas_dir = os.path.join(base_dir, agency, "structured_agendas")
		for filename in os.listdir(agendas_dir):
			if filename.endswith(".json"):

				# load agenda as json
				filepath = os.path.join(agendas_dir, filename)
				print filepath
				with open(filepath) as data_file:
					agenda = json.load(data_file)

				# flatten json to pandas DF
				df = pd.io.json.json_normalize(data=agenda, 
						record_path=['meeting_sections', 'items'], 
						meta=['agency', 'meeting_date', ['meeting_sections','section_number'], ['meeting_sections','section_name']])
				if not df.empty:
					df_list.append(df)

	all_items_df = pd.concat(df_list, ignore_index=True)

	hand_matched_filepath = "data/training_data.csv"
	if os.path.exists(hand_matched_filepath):
		hand_matched_df = pd.read_csv(hand_matched_filepath, index_col='match_id')
		print(hand_matched_df.head())

	return pd.concat(df_list, ignore_index=True)
	

'''
matchReportsToItems
===================
Given a dataframe of reports and a df of agenda items, add outcome codes from the reports
to the appropriate items.
Return the classified df.
'''
def matchReportsToItems(reports_df, items_df, parsed_agencies):

	# add columns to reports df
	reports_df['matched'] = False

	# add columns to items_df
	items_df['priority_wpusa'] = None
	items_df['priority_sblc'] = None
	items_df['priority_ibew'] = None
	items_df['priority_unite'] = None
	items_df['match_confirmed'] = False
	generateItemIDs(items_df)

	for i, row in reports_df.iterrows():

		# get agency code
		agency_name = row['agency']
		agency_id_list = [agency['agency_id'] for agency in parsed_agencies if agency_name in agency['aliases']]
		
		# make sure this agency has been parsed
		if not agency_id_list:
			continue

		agency_id = agency_id_list[0]

		# get other identifying information
		if not len(row['meeting_date']):
			print "MISSING DATE"
			print row
			continue

		# check if this includes multiple item numbers
		if re.search(r',|\+', row['item_number']):
			item_numbers = re.split(r',\s?|\+\s?', row['item_number'])
			for item_num in item_numbers:
				matchItem(agency_id, row['meeting_date'], item_num, row, items_df)
		elif row['item_number']:
			matchItem(agency_id, row['meeting_date'], row['item_number'], row, items_df)

		else:
			# try to match using the item name
			matchItem(agency_id, row['meeting_date'], None, row, items_df)


'''
generateItemIDs
===============
Uses the agency and date to generate a unique ID code for each agenda item.
'''
def generateItemIDs(items_df):
	items_df['match_id'] = None

	# position vars
	cur_item_num = 0001
	cur_agency = items_df.ix[0, 'agency']
	cur_date = items_df.ix[0, 'meeting_date']

	for i, row in items_df.iterrows():
		if row['agency'] != cur_agency or row['meeting_date'] != cur_date:
			cur_agency = row['agency']
			cur_date = row['meeting_date']
			cur_item_num = 1

		match_id = "%s_%s_%04d" % (row['agency'], row['meeting_date'], cur_item_num)
		items_df.set_value(i, 'match_id', match_id)

		cur_item_num += 1

'''
matchItem
=========
Attempts to match a report row with a row in items_df. 
If successful, sets the hand-coded priority values for that row.
'''
def matchItem(agency_id, meeting_date, item_number, row, items_df):

	# subset to the agency and meeting date
	meeting_df = items_df[(items_df['agency'] == agency_id) & (items_df['meeting_date'] == meeting_date)]
	if len(meeting_df) == 0:
		print("ERROR: could not find agenda for %s on %s" % (agency_id, meeting_date))
		print row
		return

	print("-------") # separator line
	print("%s, %s, %s" %(agency_id, meeting_date, item_number))

	# create a cleaned item name
	# clean_item_name = re.sub(r'\+', 'on', row['item_name'])
	clean_item_name = re.sub(r'\W+', ' ', row['item_name']).lower()

	# some item name strings have multiple items separated with semicolons
	item_name_list = re.split(r';\s*', row['item_name'])
	clean_item_name_list = []
	for name_comp in item_name_list:
		clean_item_name_list.append(re.sub(r'\W+', ' ', name_comp).lower())

	# if len(item_name_list) <= 1:
	# 	# some item name strings have multiple sentences that are actually different items - NOT WORKING
	# 	item_name_list = sent_tokenize(re.sub(r'\+', 'on', row['item_name']))

	# try matching with the item number and item name
	match = meeting_df[(meeting_df['item_number'] == item_number) & (meeting_df['item_text'].str.contains(row['item_name']))]
	if item_number is not None and not match.empty:
		print("Match on name and number")

	# try matching on the item name
	elif not meeting_df[meeting_df['item_text'].str.contains(row['item_name'])].empty:
		match = meeting_df[meeting_df['item_text'].str.contains(row['item_name'])]
		print("Match on item name")

	# try matching on a sanitized item name
	elif not meeting_df[meeting_df['item_text'].str.lower().str.replace(r'\W+', ' ').str.contains(clean_item_name)].empty:
		match = meeting_df[meeting_df['item_text'].str.lower().str.replace(r'\W+', ' ').str.contains(clean_item_name)]
		print("Match on cleaned item name")
		print("REPORT ITEM NAME: %s" % row['item_name'])
		if len(match['item_text']) == 1:
			print("AGENDA ITEM NAME: %s" % match['item_text'].item())
		else:
			print("AGENDA ITEM NAME: %s" % match['item_text'])

	# try matching on any of the components of the item name
	elif len(item_name_list) > 1 and not meeting_df[meeting_df['item_text'].isin(item_name_list)].empty:
		match = meeting_df[meeting_df['item_text'].isin(item_name_list)]
		print("Match on item name component")
	
	# try matching on any of the components of the cleaned item name
	elif len(clean_item_name_list) > 1 and not meeting_df[meeting_df['item_text'].str.lower().str.replace(r'\W+', ' ').isin(clean_item_name_list)].empty:
		match = meeting_df[meeting_df['item_text'].str.lower().str.replace(r'\W+', ' ').isin(clean_item_name_list)]
		print("Match on cleaned item name component")

	# try matching on the item number
	elif item_number is not None and not meeting_df[meeting_df['item_number'] == item_number].empty:
		match = meeting_df[meeting_df['item_number'] == item_number]
		print("Match on item number")

		# print stuff for human match confirmation
		print("REPORT ITEM NAME: %s" % row['item_name'])
		if len(match['item_text']) == 1:
			print("AGENDA ITEM NAME: %s" % match['item_text'].item())
		else:
			print("AGENDA ITEM NAME: %s" % match['item_text'])

	else:
		print("ERROR: Could not find match")
		print(row)
		# print("Agency: %s, Date: %s, Item: %s" % (agency_id, meeting_date, row['item_name']))
		return


	# get the match_id
	match_id = match.iloc[0]['match_id']

	# set the values in the full df
	items_df.loc[items_df['match_id'] == match_id, 'priority_wpusa'] = row['priority_wpusa']
	items_df.loc[items_df['match_id'] == match_id, 'priority_sblc'] = row['priority_sblc']
	items_df.loc[items_df['match_id'] == match_id, 'priority_ibew'] = row['priority_ibew']
	items_df.loc[items_df['match_id'] == match_id, 'priority_unite'] = row['priority_unite']


'''
writeTrainingDataToDisk
=======================
Subsets the items DF to the time period covered by the reports, 
then writes out the DF to disk.
'''
def writeTrainingDataToDisk(items_df):

	# subset to items after the start date
	start_date = "01-01-2013" # change this if expand dataset
	items_df['meeting_date'] = pd.to_datetime(items_df['meeting_date'], format='%m-%d-%Y')
	covered_items_df = items_df[items_df['meeting_date'] >= start_date]

	# write to disk
	covered_items_df.to_csv("data/training_data.csv", encoding="utf-8", index=False)












if __name__ == '__main__':
    main()