import pandas
import numpy

# column names in the input data file 
CNAME_CLASS = "TAKIMKODU"
CNAME_SUBJECT = "OZETBASLIK"
CNAME_DESCRIPTION = "ACIKLAMA"
CNAME_SUBJECT_DESCRIPTION = "OZETBASLIK_ACIKLAMA"
CNAME_RECORD_TYPE="KAYITTIPI"
CNAME_STATUS = "DURUM"
CNAME_YEAR_OPENED = "OLUSTURULDUYIL"
CNAME_MONTH_OPENED = "OLUSTURULDUAY"

# the issue assigned to a team should occur at least MIN_NUMBER_OF_DISTINCT_VALUES  times for training
MIN_NUMBER_OF_DISTINCT_VALUES = 10

# filtering specifications 
FILTER_ISSUE_TYPE = 'Issue'
FILTER_ISSUE_STATUS = 'Closed'
FILTER_ISSUE_YEAR = 2017
FILTER_ISSUE_MONTH_1 = 'JUN'
FILTER_ISSUE_MONTH_2 = 'JUL'
# ...
FILTER_ISSUE_MONTH_6 = 'NOV'
FILTER_ISSUE_YEAR_TEST = 2017
FILTER_ISSUE_MONTH_TEST = 'DEC'

def selectRecordsOpenedAtYearMonth(dataset, year, month):
    """

    returns the issue records belonging to the input year and month

    """
    d1 = dataset.groupby(CNAME_YEAR_OPENED).filter(lambda x: x.name == year)

    return d1.groupby(CNAME_MONTH_OPENED).filter(lambda x: x.name == month)

class DataLoader(object):

    def load(self, filePath):
        '''
        load the dataset, ISO-8859-9 encoding is used for characters specific to Turkish language.
        '''
        dataset = pandas.read_csv(filePath, encoding="ISO-8859-9", delimiter=";") 

        # remove the spaces from the start and end of column names
        dataset.rename(columns=lambda x: x.strip(), inplace=True)

        # concatenate subject and description in one column
        dataset[CNAME_SUBJECT_DESCRIPTION] = dataset[CNAME_SUBJECT].astype(str) + ' ' + dataset[CNAME_DESCRIPTION].astype(str)

        return dataset

class DataFilterer(object):

    def selectTrainingDatasetRecords(self, dataset):
        """
        filter issue records from the training dataset such that 
         * unresolved are eliminated and 
         * they are opened at specific time intervals
        
        """
        dataset = dataset[(dataset[CNAME_RECORD_TYPE] == FILTER_ISSUE_TYPE) &
                          (dataset[CNAME_STATUS] == FILTER_ISSUE_STATUS)]

        # # select year and month
        frames = [selectRecordsOpenedAtYearMonth(dataset, FILTER_ISSUE_YEAR, FILTER_ISSUE_MONTH_1),
                  selectRecordsOpenedAtYearMonth(dataset, FILTER_ISSUE_YEAR, FILTER_ISSUE_MONTH_2),
                  # .....
                  selectRecordsOpenedAtYearMonth(dataset, FILTER_ISSUE_YEAR, FILTER_ISSUE_MONTH_6)]
        dataset = pandas.concat(frames)
        return dataset

    def selectTestDatasetRecords(self, dataset):
        """
        filter issue records from the test dataset such that 
         * unresolved are eliminated and 
         * they are opened at specific time intervals
         
        """
        # select year and month
        frames = [selectRecordsOpenedAtYearMonth(dataset, FILTER_ISSUE_YEAR_TEST, FILTER_ISSUE_MONTH_TEST)]
        dataset = pandas.concat(frames)
        return dataset = dataset[(dataset[CNAME_RECORD_TYPE] == FILTER_ISSUE_TYPE) &
                                 (dataset[CNAME_STATUS] == FILTER_ISSUE_STATUS)]
    
    def selectRecordsHavingAtLeastNValuesInColumn(self, dataset, columnName):
        """
        returns records that have the same value at columnName at least N times 

        """

        return dataset.groupby(columnName).filter(lambda x: len(x) >= MIN_NUMBER_OF_DISTINCT_VALUES)
