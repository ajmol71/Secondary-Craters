import pandas as pd
import numpy as np

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import math
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import _tree
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import xgboost

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

def get_acc(m_name, mdl, x, y):
    pred = mdl.predict(x)
    prob = mdl.predict_proba(x)

    acc = metrics.accuracy_score(y, pred)
    p1 = prob[:, 1]

    fpr, tpr, threshold = metrics.roc_curve(y, p1)
    auc = metrics.auc(fpr, tpr)

    stats = [m_name, mdl, acc, fpr, tpr, auc]


    return stats

def add_auc(m_name, df, metrics):
    index = len(df)
    df.loc[index] = metrics
    df = df.rename(index={index: m_name})
    return df

# LOAD DATA
all_data = pd.read_csv("all_data_imputed.csv", index_col=0)
all_data = all_data.drop([x for x in all_data.columns if "NOTES" in x], axis=1)

y_raw = all_data["SECONDARY"]

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)

crater_ids = all_data["CRATER_ID"]

og_data = all_data.iloc[:, 1:32]
imp_data = all_data.iloc[:, 33:]
x = imp_data.copy()

categoricals = ["LAY_NUMBER", "LAY_MORPH1", "LAY_MORPH2", "LAY_MORPH3", "INT_MORPH1", "INT_MORPH2", "INT_MORPH3", "DEG_RIM", "DEG_EJC", "DEG_FLR", "CONF"]
numericals = ["LAT_CIRC_IMG", "LON_CIRC_IMG", "LAT_ELLI_IMG", "LON_ELLI_IMG", "DIAM_CIRC_IMG", "DIAM_CIRC_SD_IMG", "DIAM_ELLI_MAJOR_IMG", "DIAM_ELLI_MINOR_IMG", "DIAM_ELLI_ECCEN_IMG", "DIAM_ELLI_ELLIP_IMG", "DIAM_ELLI_ANGLE_IMG", "DIAM_ELLI_ANGLE_IMG", "LAT_ELLI_SD_IMG", "LON_ELLI_SD_IMG", "DIAM_ELLI_MAJOR_SD_IMG", "DIAM_ELLI_MINOR_SD_IMG", "DIAM_ELLI_ANGLE_SD_IMG", "DIAM_ELLI_ECCEN_SD_IMG", "DIAM_ELLI_ELLIP_SD_IMG", "ARC_IMG", "PTS_RIM_IMG"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .75, test_size = .25, random_state=1)
vars_df = pd.DataFrame(columns = x_train.columns)
aucs = pd.DataFrame(columns = ["m_name", "mdl", "acc", "fpr", "tpr", "auc"])

def make_model(mdl_name, mdl, aucs):
    mdl = mdl.fit(x_train, y_train)

    metrs_tr = get_acc(mdl_name + "_train", mdl, x_train, y_train)
    metrs_te = get_acc(mdl_name + "_test", mdl, x_test, y_test)

    aucs = add_auc(mdl_name, aucs, metrs_tr)
    aucs = add_auc(mdl_name, aucs, metrs_te)
    pickle_model(mdl_name, mdl)
    return mdl, aucs

def pickle_model(m_name, mdl):
    pickle.dump(mdl, open(m_name + ".pkl", 'wb'))
    return

def load_model(m_name):
    return pickle.load(open(m_name + ".pkl", "rb"))

def tree_vars(m_name, mdl, type, vars_df):
    if type == "train":
        vars = list(x_train.columns.values)
    elif type == "test":
        vars =list(x_test.columns.values)
    else:
        pass

    tree_ = mdl.tree_
    # var_list = [vars[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    nameSet = set()
    for i in tree_.feature:
        if i != _tree.TREE_UNDEFINED:
            nameSet.add(i)
    nameList = list(nameSet)
    parameter_list = list()
    for i in nameList:
        parameter_list.append(vars[i])

    df_list = []
    for j in x_train.columns:
        if j in parameter_list:
            df_list.append(1)
        else:
            df_list.append(0)

    index = len(vars_df)
    vars_df.loc[index] = df_list
    vars_df = vars_df.rename(index={index: m_name})

    return parameter_list, vars_df

def ensemble_vars(m_name, mdl, type, vars_df):
    if type == "train":
        vars = list(x_train.columns.values)
    elif type == "test":
        vars =list(x_test.columns.values)
    else:
        pass

    importance = mdl.feature_importances_
    index = np.argsort(importance)
    theList = []
    for i in index:
        imp_val = importance[i]
        if imp_val > np.average(mdl.feature_importances_):
            v = int(imp_val / np.max(mdl.feature_importances_) * 100)
            theList.append((vars[i], v))
    ens_vars = sorted(theList, key=itemgetter(1), reverse=True)

    df_list = []
    for j in x_train.columns:
        if j in ens_vars:
            df_list.append(1)
        else:
            df_list.append(0)

    index = len(vars_df)
    vars_df.loc[index] = df_list
    vars_df = vars_df.rename(index={index: m_name})

    return ens_vars, vars_df



# DECISION TREE
mdl_name = "dec_tree"
mdl = tree.DecisionTreeClassifier(max_depth = 4)

dec_tree, aucs = make_model(mdl_name, mdl, aucs)
dec_train_vars, vars_df = tree_vars(mdl_name, mdl, "train", vars_df)
dec_test_vars, vars_df = tree_vars(mdl_name, mdl, "test", vars_df)

pickle_model(mdl_name, mdl)


# RANDOM FOREST
mdl_name = "rand_for"
mdl = RandomForestClassifier(max_depth = 4)

rand_for, aucs = make_model(mdl_name, mdl, aucs)
rand_train_vars, vars_df = ensemble_vars(mdl_name, mdl, "train", vars_df)
rand_test_vars, vars_df = ensemble_vars(mdl_name, mdl, "test", vars_df)

pickle_model(mdl_name, mdl)


# GRAD BOOST
mdl_name = "grad_boost"
mdl = GradientBoostingClassifier(max_depth = 4)

grad_b, aucs = make_model(mdl_name, mdl, aucs)
grad_train_vars, vars_df = ensemble_vars(mdl_name, mdl, "train", vars_df)
grad_test_vars, vars_df = ensemble_vars(mdl_name, mdl, "test", vars_df)

pickle_model(mdl_name, mdl)


aucs_df = pd.DataFrame(aucs, columns = ["m_name", "mdl", "acc", "fpr", "tpr", "auc"])
vars_df = pd.DataFrame(vars_df, columns = x_train.columns)
aucs_df.to_csv("aucs_df.csv")
vars_df.to_csv("vars_df.csv")