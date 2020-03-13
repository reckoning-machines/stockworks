import quandl
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from scipy import stats
import streamlit as st
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV

scaler = StandardScaler()

Q_KEY = "_uccF2ycfazsfESzVDzW"

stock_ticker = 'STT' #ticker for analysis vs S&P 500
end_date = "2020-03-05" #end date of stock prices

tickers = ["^GSPC",
           stock_ticker]

st.title("Fit model for ticker: {0:s}".format(stock_ticker))

#helper function to apply yoy and qoq transformation to Quandl data
def change_columns(data,col_list,sort_column):
    data = data.sort_values(sort_column)
    for col in col_list:
        data[col+'_qoq'] = data[col].pct_change(1)
        data[col+'_yoy'] = data[col].pct_change(4)
    data = data.sort_values(sort_column,ascending = False)
    return data

#pull quandl data
quandl.ApiConfig.api_key = Q_KEY
data = quandl.get_table('SHARADAR/SF1', ticker=stock_ticker, paginate=True)

#filter dataset down to quarterly and specific columns
data = data[['ticker','dimension','price','dps','calendardate','datekey','reportperiod','payoutratio','bvps','epsdil','roe','roa','de','revenue','netinccmn','assets','equity']]
data = data[data['dimension']=='ARQ']

#annualize ratios
data['roe'] = (data['netinccmn'] / data['equity']) *4
data['roa'] = (data['netinccmn'] / data['assets']) *4
data['turnover'] = (data['revenue'] / data['assets']) *4
data['roa'] = (data['netinccmn'] / data['assets']) *4

#apply qoq and yoy to specific columns
list_change_cols = ['turnover','assets','payoutratio','bvps','epsdil','roe','roa','de','revenue','dps']
data = change_columns(data,list_change_cols,'calendardate')

#get stock price data for stock and S&P 500
multpl_stocks = web.get_data_yahoo(tickers,start = "2005-03-31",end = end_date)
daily_returns = multpl_stocks['Adj Close'].pct_change().reset_index()

daily_returns = daily_returns.fillna(0)
daily_returns['index'] = daily_returns.index
df = daily_returns[['^GSPC',stock_ticker]]
endog = daily_returns[stock_ticker]
exog = sm.add_constant(daily_returns['^GSPC'])
rols = RollingOLS(endog, exog, window=60)
rres = rols.fit()
df_rolling = rres.params
df_rolling['index'] = df_rolling.index
df_rolling['rolling_beta'] = df_rolling['^GSPC']
df_rolling = df_rolling.drop(columns = ['const','^GSPC'])

daily_merged = pd.merge(df_rolling,daily_returns,how = 'inner',left_on='index',right_on = 'index')
daily_merged['beta_return'] = daily_merged['^GSPC'] * daily_merged['rolling_beta']
daily_merged['alpha'] = daily_merged['beta_return'] - daily_merged[stock_ticker]

daily_merged = daily_merged.dropna()
daily_merged['zscore'] = stats.zscore(daily_merged['alpha'])
daily_merged['Date'] = pd.to_datetime(daily_merged['Date'])
daily_merged.set_index('Date', inplace=True)
daily_merged = daily_merged.resample('M').mean()
daily_merged['flag']  = daily_merged['zscore'].apply(lambda x: 1 if x > 0 else 0)

data['reportperiod'] = pd.to_datetime(data['reportperiod'])
data.set_index('reportperiod', inplace=True)
data['quarter'] = data['calendardate'].dt.quarter
data = data.resample('M').first()
data = data.fillna(method='ffill')


a = daily_merged.reset_index()
b = data.reset_index()
b['Date'] = b['reportperiod']
fit_data = pd.merge(a,b)
fit_data = fit_data.dropna()

feature_cols = ['assets_qoq', 'assets_yoy','turnover_qoq', 'turnover_yoy', 'payoutratio_qoq',
       'payoutratio_yoy', 'bvps_qoq', 'bvps_yoy', 'epsdil_qoq', 'epsdil_yoy',
       'roe_qoq', 'roe_yoy', 'roa_qoq', 'roa_yoy', 'de_qoq', 'de_yoy',
       'revenue_qoq', 'revenue_yoy', 'dps_qoq', 'dps_yoy', 'quarter']

X = fit_data.loc[:, feature_cols]
y = fit_data.flag

scaled_features = StandardScaler().fit_transform(X.values)
scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
X = scaled_features_df

clf = LogisticRegression()
clf.fit(X,y)

fit_score = clf.score(X,y)

i = 0
dict_features = {}
for item in clf.coef_[0]:
    dict_features[X.columns[i]]=item
    i = i+1

feat_imp = pd.Series(dict_features).sort_values(ascending=False)

df_features = pd.DataFrame(feat_imp,columns=['score']).reset_index()

st.write("LogisticRegression model score (Train): {a:8.2f}".format(a=fit_score))

st.write("LogisticRegression Feature Importance:")
st.write(alt.Chart(df_features).mark_bar().encode(
    x=alt.X('index', sort=None),
    y='score',
))

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['flag'],eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

    #Print model report:
    st.write("XGBoost Accuracy : %.2g" % metrics.accuracy_score(dtrain['flag'].values, dtrain_predictions))
    st.write("XGBoost AUC Score (Train): %.2g" % metrics.roc_auc_score(dtrain['flag'], dtrain_predprob))

    dict_gain = alg.get_booster().get_score(importance_type= 'gain')
    feat_imp = pd.Series(dict_gain).sort_values(ascending=False)
    df_features = pd.DataFrame(feat_imp,columns=['score']).reset_index()

    st.write("XGBoost Feature Importance:")
    st.write(alt.Chart(df_features).mark_bar().encode(
        x=alt.X('index', sort=None),
        y='score',
    ))

#    print("\nModel Report")
#    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['flag'].values, dtrain_predictions))

#Choose all predictors except target & IDcols
feature_cols = ['flag','assets_qoq', 'assets_yoy','turnover_qoq', 'turnover_yoy', 'payoutratio_qoq',
       'payoutratio_yoy', 'bvps_qoq', 'bvps_yoy', 'epsdil_qoq', 'epsdil_yoy',
       'roe_qoq', 'roe_yoy', 'roa_qoq', 'roa_yoy', 'de_qoq', 'de_yoy',
       'revenue_qoq', 'revenue_yoy', 'dps_qoq', 'dps_yoy', 'quarter']
train = fit_data.loc[:, feature_cols]
target = 'flag'
predictors = [x for x in train.columns if x not in [target]]

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


st.write("Data")
feature_cols.append('flag')
data = fit_data.loc[:, feature_cols]
st.write(data)
