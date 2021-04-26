#---------------------------------------------------------------------------------------------------
# This is the main file for the ML project (John Bryce course 7718-5)
# Written by Oved Dahari, December 2020.
# In this version, the Random Forest model is used.
# Three models may be selected by the user: high precision, optimal, and high recall.
# This version enables the user to test the models on classified data.
# Version 1.9
#---------------------------------------------------------------------------------------------------

# load libraries
import SDSS_aux.sdss_oveds_accs as acc
import pandas as pd

print('\nWelcome to the QSO-search tool, using the SDSS database')
print('Written by \xa9Oved_Dahari, Version 1.8 (January 2021)')

# check if model files are on disk. If not - train the models.
models_exists = True
for ii in range(1,4):
    try:
        file_name = 'SDSS_aux\SDSS_RF_Model' + str(ii) + '.p'
        with open(file_name, 'rb') as file:
            continue
    except:
        models_exists = False
        break
if not models_exists:
    acc.train_RF_models()

# The user may want to test the models first:
is_test = acc.yes_or_no('\nWould you like to test the models first? (y/n): ')
while is_test == 'y':
    acc.test_models()
    print('\nThe expected precision and recall are as follows:')
    print(' model             precision     recall')
    print('---------------------------------------------------')
    print('high precision     0.92          0.54')
    print('best accuracy      0.82          0.78')
    print('high recall        0.69          0.89')
    print('uncertainties are ~2% for precision, and ~3% for recall')
    is_test = acc.yes_or_no('Another test? ')

# The main prediction, may be run for many regions (each creates a separate output file)
print('\nNext, select the sample to predict')

another_region = True
while another_region:

    # The user selects a region in the sky to retrieve data from
    # If no data is found (sky region not covered by SDSS) - try again
    region = [acc.select_region('photoObj')]

    found = False
    search_data = pd.DataFrame()
    while not found:
        search_data = acc.load_raw(region, 'photoObj')
        if search_data.shape[0] == 0:
            print('\nSky portion not covered by SDSS, select another region\n')
            region = acc.select_region('photoObj')
        else:
            found = True

    print("\nYour region contains", search_data.shape[0], "objetcs")

    # The user selects an option for the model, and may run other models from the same region
    another_model = True
    while another_model:
        model_option = acc.select_option()
        acc.find_QSOs(search_data, model_option)
        if acc.yes_or_no('\nTry another option (same sky region)? (y/n): ') == 'n':
            another_model = False

    if acc.yes_or_no('\nTry another region in the sky? (y/n): ') == 'n':
        another_region = False

print('\nDone... Thank You')

#----------------------------------That's It-------------------------------------