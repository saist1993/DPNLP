#!/usr/bin/env python
import os
from collections import defaultdict
from random import shuffle
import pandas as pd
import numpy as np
from pathlib import Path


folder_location = '../../datasets'



def get_celeb_data(load_data_size=None):
    """Load the celebA dataset.
    Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    # src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'celebA/list_attr_celeba.csv'), sep=';')
    df = df.rename(columns={'Male': 'sex'})

    s = -1 * df['sex']
    y = df['Smiling']
    # y = df['Attractive']
    df = df.drop(columns=['sex', 'Smiling', 'picture_ID'])
    # df = df.drop(columns=['sex', 'Attractive', 'picture_ID'])

    X = df.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    # remove duplicates
    # _, unique_indices = np.unique(X, axis=0, return_index=True)

    # return X[unique_indices,:], y[unique_indices], s[unique_indices]
    return X, y, s


def get_celeb_multigroups_data():
    """Load the celebA dataset.
    Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    # src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'celebA/list_attr_celeba.csv'), sep=';')
    df = df.rename(columns={'Male': 'sex'})

    s1 = -1 * df['sex']
    s2 = -1 * df['Young']
    y = df['Smiling']
    df = df.drop(columns=['sex', 'Young', 'Smiling', 'picture_ID'])

    X = df.to_numpy()
    y = y.to_numpy()
    s1 = s1.to_numpy()
    s2 = s2.to_numpy()

    X = X[:, (X != 0).any(axis=0)]

    _, s = np.unique(np.hstack((s1.reshape(-1, 1), s2.reshape(-1, 1))), return_inverse=True, axis=0)

    return X, y, s


def get_adult_data(load_data_size=None):
    """Load the Adult dataset.
    Source: UCI Machine Learning Repository.

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    def mapping(tuple):
        # native country
        tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"
        # education
        if tuple['education'] in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
            tuple['education'] = "prim-middle-school"
        if tuple['education'] in ["9th", "10th", "11th", "12th"]:
            tuple['education'] = "high-school"
        return tuple

    # src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'adult/adult.csv'))
    df = df.drop(['race'], axis=1)
    df = df.replace("?", np.nan)
    df = df.dropna()
    df = df.apply(mapping, axis=1)

    sensitive_attr_map = {'Male': 1, 'Female': -1}
    label_map = {'>50K': 1, '<=50K': -1}

    attrs = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    int_attrs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
                 'fnlwgt']  # we need to add fnlwgt, otherwise we have too many duplicates

    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['income'].map(label_map).astype(int)

    x = pd.DataFrame(data=None)
    for x_var in attrs:
        x = pd.concat([x, pd.get_dummies(df[x_var], prefix=x_var, drop_first=False)], axis=1)
    for x_var in int_attrs:
        x = pd.concat([x, normalize(x=df[x_var])], axis=1)

    X = x.to_numpy()
    s = s.to_numpy()
    y = y.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    # remove duplicates
    # _, unique_indices = np.unique(X, axis=0, return_index=True)

    # return X[unique_indices,:], y[unique_indices], s[unique_indices]
    return X, y, s


def get_adult_multigroups_data(load_data_size=None):
    """Load the Adult dataset.
    Source: UCI Machine Learning Repository.

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """
    def mapping(tuple):
        # native country
        tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"
        # education
        if tuple['education'] in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
            tuple['education'] = "prim-middle-school"
        if tuple['education'] in ["9th", "10th", "11th", "12th"]:
            tuple['education'] = "high-school"
        tuple['race'] = 'NonWhite' if tuple['race'] != "White" else 'White'
        return tuple

    # src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'adult/adult.csv'))
    # df = df.drop(['race' ], axis=1)
    df = df.replace("?", np.nan)
    df = df.dropna()
    df = df.apply(mapping, axis=1)

    sensitive_attr_map1 = {'Female': 1, 'Male': -1}
    sensitive_attr_map2 = {'NonWhite': 1, 'White': -1}
    label_map = {'>50K': 1, '<=50K': -1}

    attrs = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    int_attrs = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt']

    s1 = df['sex'].map(sensitive_attr_map1).astype(int)
    s2 = df['race'].map(sensitive_attr_map2).astype(int)
    y = df['income'].map(label_map).astype(int)

    x = pd.DataFrame(data=None)
    for x_var in attrs:
        x = pd.concat([x, pd.get_dummies(df[x_var], prefix=x_var, drop_first=False)], axis=1)
    for x_var in int_attrs:
        x = pd.concat([x, normalize(x=df[x_var])], axis=1)

    X = x.to_numpy()
    s1 = s1.to_numpy()
    s2 = s2.to_numpy()
    y = y.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s1 = s1[perm]
        s2 = s2[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s1 = s1[:load_data_size]
        s2 = s2[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    # remove duplicates
    # _, unique_indices = np.unique(X, axis=0, return_index=True)

    # return X[unique_indices,:], y[unique_indices], s1[unique_indices], s2[unique_indices]
    _, s = np.unique(np.hstack((s1.reshape(-1, 1), s2.reshape(-1, 1))), return_inverse=True, axis=0)

    return X, y, s


def get_adult_multigroups_data_sensr(load_data_size=None):
    """

    This adult multigroup dataset is a different than the original adult multigroup data.
    In the original one there are 52 features while in this one there are 41 features.

    The implementation is from the paper: Training individual Fair ML models with sensitive subspace robustness
    reference implementation: https://github.com/IBM/sensitive-subspace-robustness
    """

    from aif360.datasets import BinaryLabelDataset
    from sklearn.preprocessing import OneHotEncoder, StandardScaler



    def mapping(tuple):
        # native country
        # tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"
        # education
        # if tuple['education'] in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
        #     tuple['education'] = "prim-middle-school"
        # if tuple['education'] in ["9th", "10th", "11th", "12th"]:
        #     tuple['education'] = "high-school"
        tuple['race'] = 'NonWhite' if tuple['race'] != "White" else 'White'
        return tuple


    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']
    train = pd.read_csv(os.path.join(folder_location, 'adult_sensr/adult.data'), header = None)
    test = pd.read_csv(os.path.join(folder_location, 'adult_sensr/adult.test'), header = None)
    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({' <=50K.': 0, ' >50K.': 1, ' >50K': 1, ' <=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    delete_these = ['race_ Amer-Indian-Eskimo','race_ Asian-Pac-Islander','race_ Black','race_ Other', 'sex_ Female']

    delete_these += ['native-country_ Cambodia', 'native-country_ Canada', 'native-country_ China', 'native-country_ Columbia', 'native-country_ Cuba', 'native-country_ Dominican-Republic', 'native-country_ Ecuador', 'native-country_ El-Salvador', 'native-country_ England', 'native-country_ France', 'native-country_ Germany', 'native-country_ Greece', 'native-country_ Guatemala', 'native-country_ Haiti', 'native-country_ Holand-Netherlands', 'native-country_ Honduras', 'native-country_ Hong', 'native-country_ Hungary', 'native-country_ India', 'native-country_ Iran', 'native-country_ Ireland', 'native-country_ Italy', 'native-country_ Jamaica', 'native-country_ Japan', 'native-country_ Laos', 'native-country_ Mexico', 'native-country_ Nicaragua', 'native-country_ Outlying-US(Guam-USVI-etc)', 'native-country_ Peru', 'native-country_ Philippines', 'native-country_ Poland', 'native-country_ Portugal', 'native-country_ Puerto-Rico', 'native-country_ Scotland', 'native-country_ South', 'native-country_ Taiwan', 'native-country_ Thailand', 'native-country_ Trinadad&Tobago', 'native-country_ United-States', 'native-country_ Vietnam', 'native-country_ Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    dataset_orig =  BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=['sex_ Male', 'race_ White'])

    # we will standardize continous features
    continous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    continous_features_indices = [dataset_orig.feature_names.index(feat) for feat in continous_features]
    SS = StandardScaler().fit(dataset_orig.features[:, continous_features_indices])
    dataset_orig.features[:, continous_features_indices] = SS.transform(
        dataset_orig.features[:, continous_features_indices])

    X = dataset_orig.features
    y = dataset_orig.labels

    one_hot = OneHotEncoder(sparse=False)
    one_hot.fit(y.reshape(-1,1))
    names_income = one_hot.categories_
    y = np.argmax(one_hot.transform(y.reshape(-1,1)), axis=1)

    s_gender = dataset_orig.features[:,dataset_orig.feature_names.index('sex_ Male')]

    s_race = dataset_orig.features[:, dataset_orig.feature_names.index('race_ White')]

    s_concat, s = np.unique(np.hstack((s_gender.reshape(-1, 1), s_race.reshape(-1, 1))), return_inverse=True, axis=0)

    return X, y, s






def load_adult_data_zafar(load_data_size=None):
    """
        if load_data_size is set to None (or if no argument is provided), then we load and return the whole data
        if it is a number, say 10000, then we will return randomly selected 10K examples
    """

    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country']  # all attributes
    int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss',
                 'hours_per_week']  # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['sex']  # the fairness constraints will be used for this feature
    attrs_to_ignore = ['sex', 'race',
                       'fnlwgt']  # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
    data_files = ["adult.data", "adult.test"]

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {}  # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:

        for line in open(f):
            line = line.strip()
            if line == "": continue  # skip empty lines
            line = line.split(", ")
            if len(line) != 15 or "?" in line:  # if a line has missing attributes, ios.path.join(folder_location,gnore it
                continue

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = -1
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)

            for i in range(0, len(line) - 1):
                attr_name = attrs[i]
                attr_val = line[i]
                # reducing dimensionality of some very sparse features
                if attr_name == "native_country":
                    if attr_val != "United-States":
                        attr_val = "Non-United-Stated"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d):  # discretize the string attributes
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs: continue
            uniq_vals = sorted(list(set(attr_vals)))  # get unique values

            # compute integer codes for the unique values
            val_dict = {}
            for i in range(0, len(uniq_vals)):
                val_dict[uniq_vals[i]] = i

            # replace the values with their integer encoding
            for i in range(0, len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country":  # the way we encoded native country, its binary now so no need to apply one hot encoding on it
            X.append(attr_vals)

        else:
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)

                # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)

    # # shuffle the data
    # perm = range(0, len(y))  # shuffle the data before creating each fold
    # shuffle(perm)
    # X = X[perm]
    # y = y[perm]
    # for k in x_control.keys():
    #     x_control[k] = x_control[k][perm]
    #
    # # see if we need to subsample the data
    # if load_data_size is not None:
    #     X = X[:load_data_size]
    #     y = y[:load_data_size]
    #     for k in x_control.keys():
    #         x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control


def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    in_arr = np.array(in_arr, dtype=int)
    assert (len(in_arr.shape) == 1)  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for i in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict


def get_crimeCommunities_data(load_data_size=None):
    """Load the Communities and Crime dataset.
    Source: UCI Machine Learning Repository

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    # src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'crimeCommunities/communities_data.csv'))

    df['ViolentCrimesPerPop'] = df['ViolentCrimesPerPop'].apply(lambda x: -1 if x <= 0.24 else 1)
    df['racePctWhite'] = df['racePctWhite'].apply(lambda x: 'other' if x <= 0.75 else 'white')
    df = df.drop(columns=['state', 'county', 'community', 'communityname string', 'fold', 'OtherPerCap',
                          # 'medIncome', 'pctWWage', 'pctWInvInc','medFamInc',
                          'LemasSwornFT', 'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop',
                          'LemasTotalReq',
                          'LemasTotReqPerPop', 'PolicReqPerOffic', 'PolicPerPop', 'RacialMatchCommPol',
                          'PctPolicWhite', 'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian', 'PctPolicMinor',
                          'OfficAssgnDrugUnits',
                          'NumKindsDrugsSeiz', 'PolicAveOTWorked', 'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
                          'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop'])
    # 29 attributes are dropped because of missing values in these features, or because they contain IDs or names

    df = df.rename(columns={'racePctWhite': 'race'})

    sensitive_attr_map = {'white': 1, 'other': -1}

    s = df['race'].map(sensitive_attr_map).astype(int)
    y = df['ViolentCrimesPerPop']

    df = df.drop(columns=['race', 'ViolentCrimesPerPop'])

    x = pd.DataFrame(data=None)
    for name in df.columns:
        x = pd.concat([x, normalize(x=df[name])], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s


def get_german_data(load_data_size=None):
    """Load the German Credit dataset.
    Source: UCI Machine Learning Repository.

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'german/german.csv'))

    sexdict = {'A91': 'male', 'A93': 'male', 'A94': 'male',
               'A92': 'female', 'A95': 'female'}
    df = df.assign(personal_status=df['personal_status'].replace(to_replace=sexdict))
    df = df.rename(columns={'personal_status': 'sex'})

    sensitive_attr_map = {'male': 1, 'female': -1}
    label_map = {1: 1, 2: -1}

    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['credit'].map(label_map).astype(int)

    x_vars_categorical = [
        'status', 'credit_history', 'purpose', 'savings', 'employment',
        'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level',
        'telephone', 'foreign_worker'
    ]

    x_vars_ordinal = [
        'month', 'credit_amount', 'investment_as_income_percentage',
        'residence_since', 'age', 'number_of_credits', 'people_liable_for'
    ]

    x = pd.DataFrame(data=None)
    for x_var in x_vars_ordinal:
        x = pd.concat([x, normalize(x=df[x_var])], axis=1)
    for x_var in x_vars_categorical:
        x = pd.concat([x, pd.get_dummies(df[x_var], prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s


def get_compas_data(load_data_size=None):
    """Load the Compas dataset.
    Source: Propublica Github repository: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'compas/propublica-recidivism.csv'))

    """Preprocessing according to https://github.com/propublica/compas-analysis"""
    df = df[(df.days_b_screening_arrest <= 30) &
            (df.days_b_screening_arrest >= -30) &
            (df.is_recid != -1) &
            (df.c_charge_degree != '0') &
            (df.score_text != 'N/A')]

    df = df.drop(columns=['is_recid', 'decile_score', 'score_text'])

    racedict = {'Caucasian': 'White', 'Other': 'NonWhite', 'African-American': 'NonWhite',
                'Hispanic': 'NonWhite', 'Asian': 'NonWhite', 'Native American': 'NonWhite'}
    df = df.assign(race=df['race'].replace(to_replace=racedict))

    sensitive_attr_map = {'White': 1, 'NonWhite': -1}
    label_map = {1: 1, 0: -1}

    s = df['race'].map(sensitive_attr_map).astype(int)
    y = df['two_year_recid'].map(label_map).astype(int)

    x_vars_categorical = ['age_cat', 'c_charge_degree', 'sex']
    x_vars_ordinal = ['age', "priors_count", 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                      'days_b_screening_arrest']

    x = pd.DataFrame(data=None)
    for x_var in x_vars_ordinal:
        x = pd.concat([x, normalize(df[x_var])], axis=1)
    for x_var in x_vars_categorical:
        x = pd.concat([x, pd.get_dummies(df[x_var], prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s


from sklearn import preprocessing


def load_compas_data_zafar():
    FEATURES_CLASSIFICATION = ["age_cat", "race", "sex", "priors_count",
                               "c_charge_degree"]  # features to be used for classification
    CONT_VARIABLES = [
        "priors_count"]  # continuous features, will need to be handled separately from categorical features, categorical features will be encoded using one-hot
    CLASS_FEATURE = "two_year_recid"  # the decision variable
    SENSITIVE_ATTRS = ["race"]

    COMPAS_INPUT_FILE = os.path.join(folder_location, "compas-scores-two-years.csv")

    # load the data and get some stats
    df = pd.read_csv(COMPAS_INPUT_FILE)
    df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals

    # convert to np array
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])

    """ Filtering the data """

    # These filters are the same as propublica (refer to https://github.com/propublica/compas-analysis)
    # If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
    idx = np.logical_and(data["days_b_screening_arrest"] <= 30, data["days_b_screening_arrest"] >= -30)

    # We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
    idx = np.logical_and(idx, data["is_recid"] != -1)

    # In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).
    idx = np.logical_and(idx, data["c_charge_degree"] != "O")  # F: felony, M: misconduct

    # We filtered the underlying data from Broward county to include only those rows representing people who had either recidivated in two years, or had at least two years outside of a correctional facility.
    idx = np.logical_and(idx, data["score_text"] != "NA")

    # we will only consider blacks and whites for this analysis
    idx = np.logical_and(idx, np.logical_or(data["race"] == "African-American", data["race"] == "Caucasian"))

    # select the examples that satisfy this criteria
    for k in data.keys():
        data[k] = data[k][idx]

    """ Feature normalization and one hot encoding """

    # convert class label 0 to -1
    y = data[CLASS_FEATURE]
    y[y == 0] = -1

    pd.Series(y).value_counts()

    X = np.array([]).reshape(len(y),
                             0)  # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)

    feature_names = []
    for attr in FEATURES_CLASSIFICATION:
        vals = data[attr]
        if attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = preprocessing.scale(vals)  # 0 mean and 1 variance
            vals = np.reshape(vals, (len(y), -1))  # convert from 1-d arr to a 2-d arr with one col

        else:  # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        # add to sensitive features dict
        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        # add to learnable features
        X = np.hstack((X, vals))

        if attr in CONT_VARIABLES:  # continuous feature, just append the name
            feature_names.append(attr)
        else:  # categorical features
            if vals.shape[1] == 1:  # binary features that passed through lib binarizer
                feature_names.append(attr)
            else:
                for k in lb.classes_:  # non-binary categorical features, need to add the names for each cat
                    feature_names.append(attr + "_" + str(k))

    # convert the sensitive feature to 1-d array
    x_control = dict(x_control)
    for k in x_control.keys():
        assert (x_control[k].shape[1] == 1)  # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()

    # sys.exit(1)

    # """permute the date randomly"""
    # perm = range(0, X.shape[0])
    # shuffle(perm)
    # X = X[perm]
    # y = y[perm]
    # for k in x_control.keys():
    #     x_control[k] = x_control[k][perm]

    return X, y, x_control


def get_dutch_data(load_data_size=None):
    """Load the Dutch Census dataset.
    Source: https://web.archive.org/web/20180108214635/https://sites.google.com/site/conditionaldiscrimination/

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(folder_location, 'dutch/dutch.csv'))

    sensitive_attr_map = {2: 1, 1: -1}
    label_map = {'5_4_9': 1, '2_1': -1}

    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['occupation'].map(label_map).astype(int)

    x_vars_categorical = [
        'household_position',
        'household_size',
        'citizenship',
        'country_birth',
        'economic_status',
        'cur_eco_activity',
        'Marital_status'
    ]

    x_vars_ordinal = [
        'age',
        'prev_residence_place',
        'edu_level'
    ]

    x = pd.DataFrame(data=None)
    for x_var in x_vars_ordinal:
        x = pd.concat([x, normalize(x=df[x_var])], axis=1)
    for x_var in x_vars_categorical:
        x = pd.concat([x, pd.get_dummies(df[x_var], prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None:  # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    # remove duplicates
    # _, unique_indices = np.unique(X, axis=0, return_index=True)

    # return X[unique_indices,:], y[unique_indices], s[unique_indices]
    return X, y, s


def normalize(x):
    # scale to [-1, 1]
    x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
    return x_


if __name__ == '__main__':
    # x, y, s = load_adult_data_zafar()
    # x, y, s = get_adult_multigroups_data_sensr()
    x, y, s = get_adult_multigroups_data()

    print(x.shape)
    print(s.shape)
    print(y.shape)
    print(s[:10])
