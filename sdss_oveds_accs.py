#-------------------------------------------------------------------------------
# This library has accessory functions for the SDSS project (JB course)
# Written by Oved Dahari on December, 2020.
# Here predict_proba is added to the output, and also a testing function.
# Only Primary objects are downloaded from SDSS tables.
# The training data is split to 8 regions (w/galactic latitude 20-90).
# The results are corrected to reflect test results
# Version 1.9
#-------------------------------------------------------------------------
import SDSS_aux.CasJobs as CasJobs       # SciServer library (loaded from GitHub)
import SDSS_aux.Authentication as au     #  - " -
import pandas as pd
from itertools import combinations as cb
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
import sklearn.metrics as skmet
import pickle

#----------------------------------------------------------------------------------
def load_raw(region_list, table):
    """
    Load raw data from a given SDSS table
    :param region_list: tuples of 3 integers (ra, dec, size)
    :param table: string
    :return: DataFrame (Pandas)
    """
        # login to the SDSS server (using SciServer libraries)
    user = 'Oved_sc84'              # Oved's username and password, as of December 2020
    pw = 'Stami101'

        # login to cloud, if fails - ask for new user and pw
    user_pw = False
    while not user_pw:
        try:
            au.login(user, pw)
            user_pw = True
        except:
            user, pw = get_user()

    print("\nDownloading data from SDSS table:", table)
    print('Please wait ~1 minute per 120k samples...')

    my_df = pd.DataFrame()

    for region in region_list:
            # compute the region's borders in degrees (and convert to strings)
        half = region[2]/2
        ra_min = str(region[0] - half)
        ra_max = str(region[0] + half)
        dec_min = str(region[1] - half)
        dec_max = str(region[1] + half)

        cls = ''                   # used for the photoObj table, which has no class
        objId = ''                 # used for the specPhoto table (no need for that info)
        if table == 'specPhoto':
            cls = ', class'             # include class for the training set
        else:
            objId = ', objID'             # include SDSS ID number for the predicted QSOs

        # prepare the SQL query (magnitudes are reduced by extinction)
        query = 'SELECT ra, dec, \
            psfmag_u - extinction_u AS mag_u, \
            psfmag_g - extinction_g AS mag_g, \
            psfmag_r - extinction_r AS mag_r, \
            psfmag_i - extinction_i AS mag_i, \
            psfmag_z - extinction_z AS mag_z, \
            z, type' + cls + objId + ' FROM ' + table + ' WHERE mode = 1 AND \
            ra BETWEEN ' + ra_min + ' AND ' + ra_max + \
            ' AND dec BETWEEN ' + dec_min + ' AND ' + dec_max

        tdf = CasJobs.executeQuery(query, 'dr16')
        # print('At ra =', region[0], ' found', tdf.shape[0], ' samples')
        my_df = pd.concat([my_df, tdf], ignore_index = True)

    print('\nTotal number of objects found:', my_df.shape[0])
    return my_df

#------------------------------------------------------------------
def get_user():
    """
    Get new username and password (at SDSS.org) from the user
    :return: string tuple
    """
    print('\nUsername and/or password not valid at SDSS.org')
    user = input('Enter new username: ')
    pw = input('Enter password: ')
    return user, pw

#-----------------------------------------------------------------
def yes_or_no(question):
    """
    Load y/n from user (check for validity)
    :param question: string
    :return: string (a single character)
    """
    while True:
        inp = input(question)
        if inp.lower() == 'y' or inp.lower() == 'n':
            return inp
        else:
            print('Please select "y" or "n"')

#-----------------------------------------------------------------
def select_option():
    """
    Get an option from user (check for validity)
    :param: none
    :return: integer (1-3)
    """
    print('\nYour options are:\n',
          '------------------\n',
          '     1. high precision (high rate of QSOs in the return sample) \n',
          '     2. optimal (best combination of precision and recall)\n',
          '     3. high recall (high rate of detection of all QSOs in the field) \n')

    while True:
        inp = input('Please enter an integer (1 to 3): ')
        if inp.isnumeric():
            ret = int(inp)
            if ret > 0 and ret < 4:
                return ret
            else:
                print('Try again...')
        else:
            print('Try again...')

#-----------------------------------------------------------------------------
def select_region(table_name):
    """
    The user selects a region in the sky to search for QSOs
    :param table_name (string)
    :return: a tuple of integers (ra, dec, size)
    """
    ra, dec, size = 0, 0, 0
    print('\nSelect a region in the sky to search')
    entry = False
    while entry == False:
        inp = input('Enter RA in degrees (0 to 359): ')
        if inp.isnumeric():
            inp = int(inp)
            if inp >= 0 and inp <= 360:
                ra = inp
                entry = True
            else:
                print('Try again...')
        else:
            print('Try again...')

    entry = False
    while entry == False:
        inp = input('Enter Dec in degrees (-20 to +80): ')
        try:
            int(inp)
        except ValueError:
            print('Try again...')
            continue
        inp = int(inp)
        if inp >= -20 and inp <= 80:
            dec = inp
            entry = True
        else:
            print('Try again...')

    entry = False
    if table_name == 'photoObj':
        print('\nNote that 1 square degree provides ~20k objects (~10% QSOs)')
    else:
        print('\nNote that 1 square degree provides ~200 objects (of which ~20% are QSOs)')
    while entry == False:
        inp = input('Enter n for nXn-square solid angle, in degrees: ')
        if inp.isnumeric():
            inp = int(inp)
            if inp > 0 and inp <= 10:
                size = inp
                entry = True
            else:
                print('Try again...')
        else:
            print('Try again...')

    return ra, dec, size

#------------------------------------------------------------------------------
def train_RF_models():
    """
    Train Random Forest Classifier models on the ~150k samples,
    Then save the models to disk (using pickle)
    :return: none
    """
    # If the training data do not exist on the disk - retrieve the data from the SDSS database
    try:
        df = pd.read_csv('SDSS_aux\SDSS_8_regions.csv')
    except:
            # list of 8 regions (galactic latitude 20-90)
        list1 = [(12, 51, 27),
            (12, 8, 30),
            (11, 22, 32),
            (10, 34, 34),
            (9, 46, 34),
            (8, 59, 32),
            (8, 14, 30),
            (7, 31, 27)]

        regions = []
        for tup in list1:
            ra = tup[0] * 15 + tup[1] // 4      # convert hh:mm to degrees (for ra)
            regions.append((ra, tup[2], 7))
                # load the data, and write to disk
        df = load_raw(regions, 'specPhoto')
        df.to_csv('SDSS_aux\SDSS_8_regions.csv')
        print('New spect-photo data file created on your disk: SDSS_aux\SDSS_8_regions.csv')

        # get the X features dataframe
    X = get_X_features(df)

        # prepare the target (QSO or not)
    temp_dict = {'QSO': 1, 'GALAXY': 0, 'STAR': 0}
    df['is_QSO'] = df['class'].replace(temp_dict)
    Y = df[['is_QSO']]

            # these weights represent the 3 different models
    weights = [{0: 10, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 10}]
    for opt in range(3):

        model = rfc(n_estimators=30, min_samples_leaf=10, max_depth=30,
                     class_weight = weights[opt], n_jobs=-1)
        print('training model', opt+1, ', please wait ~10 seconds...')
        model.fit(X, np.ravel(Y))

        out_file = 'SDSS_aux\SDSS_RF_Model' + str(opt+1) + '.p'
        with open(out_file, 'wb') as file:                           # write to a pickle file
            pickle.dump(model, file)

    print("\nModels saved on your disk in three files: 'SDSS_aux\SDSS_RF_Model(n).p'")

#-----------------------------------------------------------------------------
def find_QSOs(raw_data_in, option):
    """
    Retrieve all objects from the region, predict the QSOs, and save
    to a .csv file
    :param raw_data_in: DataFrame
    :param option: integer (1 to 3)
    :return: none
    """
    raw_data = raw_data_in.copy()
            # load the training features
    X = get_X_features(raw_data)

            # load model and run it on the data
    file_name = 'SDSS_aux\SDSS_RF_Model' + str(option) + '.p'
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    pred_p = model.predict_proba(X)[:,1].round(2)                # use predict_proba instead of predict
    raw_data.insert(4, 'predict', pred_p)                        # insert into df
                 # add features for later output
    glob_df = pd.concat([raw_data, X[['u-g', 'g-r', 'g-i', 'g-z']].round(2)], axis=1)
    glob_df[['ra', 'dec']] = glob_df[['ra', 'dec']].round(5)

    QSOs = glob_df.loc[glob_df['predict'] > 0.5]              # select only predicted QSOs
    print("Number of objects predicted as QSOs:", QSOs.shape[0])

            # print the precission and recall values from the training tests
    precission_recall = [(92, 54), (82, 78), (69, 89)]
    print("Estimated precision of the model =", precission_recall[option-1][0], '%, p/m 2%')
    print("Estimated recall of the model    =", precission_recall[option-1][1], '%, p/m 3%')

    out_features = ['objID', 'ra', 'dec', 'u-g', 'g-r', 'g-i', 'g-z', 'type', 'predict']
    out_df = QSOs[out_features]
    out_df.insert(3, 'mag_g', QSOs['mag_g'].round(2))
    out_df2 = out_df.sort_values(['ra', 'dec'])             # sort by RA, then by Dec

            # get the output filename from the user
    out_file = input('\nEnter file name to save the data to (do not include extension): ') + '.csv'
    print("\nWriting to file:", out_file)
    out_df2.to_csv(out_file, index = False)

#-----------------------------------------------------------------------------------
def test_models():
    """
    Testing the 3 models against any labeled data from specPhoto (selected by the user)
    :return: none
    """

    region = [select_region('specPhoto')]

    found = False
    data = pd.DataFrame()
    while not found:
        data = load_raw(region, 'specPhoto')
        if data.shape[0] == 0:                  # if no objects found
            print('\nSky portion not covered by SDSS, select another region\n')
            region = select_region('specPhoto')
        else:
            found = True

    print("\nYour region contains", data.shape[0], "objetcs")

    X = get_X_features(data)
    # print(X.info())

    temp_dict = {'QSO': 1, 'GALAXY': 0, 'STAR': 0}              # test QSOs vs. Others
    data['is_QSO'] = data['class'].replace(temp_dict)
    Y = data[['is_QSO']]

                # now test the 3 models
    for ii in range(1,4):
        file_name = 'SDSS_aux\SDSS_RF_Model' + str(ii) + '.p'
        with open(file_name, 'rb') as file:
            model = pickle.load(file)
        pred = model.predict(X)
        print('\nResults from model ' + str(ii) + ' (1 = QSOs)')
        print('------------------------------------')
        print(skmet.confusion_matrix(Y, pred))
        print(skmet.classification_report(Y, pred))


#-------------------------------------------------------------------------------
def get_X_features(df):
    """
    Return the X features from the raw dataframe
    :param df: DataFrame (Pandas)
    :return: DataFrame (pandas)
    """
        # convert the 'type' to 0 (extended object) or 1 (point source)
    temp_dict = {3: 0, 6: 1}
    df['is_Star'] = df['type'].replace(temp_dict)

        # prepare the training features (10 + type)
    X = pd.DataFrame(df[['is_Star']])
    for ite in cb('ugriz', 2):              # find all two-color-difference possibilities
        pair = list(ite)
        colname = pair[0] + '-' + pair[1]
        X[colname] = df['mag_' + pair[0]] - df['mag_' + pair[1]]

    return X

#-------------------------------- main (for testing) -------------------------

def main():

    # region = select_region()
    # load_raw(region, 'photoObj')

    # test_models()
    # option = select_option()
    # find_QSOs(region, option)

    # r, d, s = 192, 27, 20
    # raw_df = load_raw(r, d, s, 'SpecPhoto')
    # print('\n Number of objects found:', raw_df.shape[0])
    # print('\n first 5 are:\n', raw_df.head())
    # raw_df.to_csv('SDSS_GP400_type.csv')
    # df = pd.read_csv('SDSS_GP400_type.csv')
    train_RF_models()

if __name__ == '__main__':
    main()
