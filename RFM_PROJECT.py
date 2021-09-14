##################################################################################################################
# Libraries
##################################################################################################################

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################################################################################
########################################## RFM PROJECT WITH SALES SAMPLE DATA ####################################
##################################################################################################################

data = pd.read_csv("Ders Notları/sales_data_sample.csv", encoding='Latin-1')
df = data.copy()
df = pd.DataFrame(df)
df.head()
df.columns

def rfm_segmentation(df, analyze = False, analyze_cat = "DEALSIZE", analyze_num = "SALES"):

    # Descriptive Statistics

    print("###########################################################")
    print("READING DATA")
    print("###########################################################")
    print(df.columns)
    print("################")
    print(df.index)
    print("################")
    print(df.shape)
    print("################")
    print(df.info())
    print("################")
    print(df.describe().T)
    print("################")

    # Missing Values

    print("###########################################################")
    print("MISSING VALUE ANALYSIS")
    print("###########################################################")
    print("Is there any missing value?")
    print(df.isnull().values.any())
    missing = df.isnull().values.any()
    if (missing == True):
        print(df.isnull().sum())
    else:
        print("There is no missing value on the dataset")

    print("###########################################################")
    print("IDENTFYING VARIABLE TYPES")
    print("###########################################################")

    cat_cols = [i for i in df.columns if df[i].dtype == "O" and df[i].nunique() <= 20]

    num_but_cat = [i for i in df.columns if df[i].dtype != "O" and df[i].nunique() <= 20]

    cat_cols = cat_cols + num_but_cat

    cat_but_car = [i for i in df.columns if df[i].dtype == "O" and df[i].nunique() > 20]

    num_cols = [i for i in df.columns if df[i].dtype in [int, float] and i not in num_but_cat]



    print("Categorical Varibles: ", cat_cols)
    print("Numerical Varibles: ", num_cols)
    print("Categoric But Cardinal Variables: ", cat_but_car)

    print("###########################################################")
    print("PREPROCESSING")
    print("###########################################################")

    df = df[~df["STATUS"].isin(["Cancelled", "Disputed"])]
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    print("It has done :)")

    print("###########################################################")
    print("DATA ANALYSIS")
    print("###########################################################")

    if analyze:
        print(df[analyze_cat].value_counts())
        df[analyze_cat].value_counts().plot(kind = "bar", rot = 0)
        plt.show()

        plt.hist(df[analyze_num])
        plt.show()

        plt.boxplot(df[analyze_num])
        plt.show()
    else:
        print("You don't prefer visualizing the data")

    print("###########################################################")
    print("RFM METRICS")
    print("###########################################################")

    today_date = dt.datetime(2005, 6, 2)
    rfm = df.groupby("CUSTOMERNAME").agg({"ORDERDATE": lambda x: (today_date - x.max()).days,
                                  "ORDERNUMBER": lambda x: x.nunique(),
                                  "SALES": lambda x: x.sum()})

    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm = rfm.reset_index()

    rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4, 3, 2, 1])
    rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])
    rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], 5, labels = [1, 2, 3, 4, 5])

    rfm["RFM_SCORE"] = rfm["Recency_Score"].astype(str) + rfm["Frequency_Score"].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["SEGMENT"] = rfm["RFM_SCORE"].replace(seg_map, regex = True)

    print(rfm.groupby("SEGMENT").agg(["mean", "count"]))

rfm_segmentation(df, analyze = True)
