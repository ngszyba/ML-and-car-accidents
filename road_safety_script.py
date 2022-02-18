import pandas as pd
import numpy as np
from datetime import datetime, date
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import eli5
from eli5.sklearn import PermutationImportance
import shap 

pd.set_option('display.max_columns', None)
accidents_raw = pd.read_csv('accident-data.csv')
accidents_raw.head()

accidents_raw.info()


#add serious column giving 1 for serious accident with 3 or more casualities and 0 for other types
def what_accident(row):
    if row['number_of_casualties'] >= 3:
        return 1
    else:
        return 0
accidents['Serious'] = accidents.apply(what_accident, axis=1)
display(accidents.tail(20))

# Create a report that covers the following:
#
# 1. What time of day and day of the week do most serious accidents happen?
# 2. Are there any patterns in the time of day/ day of the week when serious accidents occur?
# 3. What characteristics stand out in serious accidents compared with other accidents?
# 4. On what areas would you recommend the planning team focus their brainstorming efforts to reduce serious accidents?


#serious accidents vs other accidents
accidents["full_hour"] = [x[:2] for x in accidents["time"].astype('str')]
days = {1:"Sunday", 2:"Monday", 3:"Tuesday", 4:"Wednesday", 5:"Thursday", 6:"Friday", 7:"Saturday"}
accidents["real_day"] = [days[x] for x in accidents["day_of_week"]]

accidents['number_of_casualties'].value_counts(sort=True).plot(kind="bar", title = 'Severity')

# +
#accidents['Serious'] = [0 if  i == 3 else 1 for i in accidents['accident_severity']]
# -

accidents.tail(10)

#What time of the day most accidents happen
by_time = accidents.groupby("full_hour")["accident_index"].count().reset_index()
ax = sns.barplot(x = "full_hour", y = "accident_index", data = by_time, palette = "Blues")
ax.set(xlabel = "hour of the day", ylabel = " Number of serious accidents")
plt.show()

#What day of the week most accidents happen
by_day_1 = accidents.groupby(["day_of_week","real_day"])["accident_index"].count().reset_index()
by_day = by_day_1.reindex([1,2,3,4,5,6,7,0])
ax1 = sns.barplot(x = "real_day", y = "accident_index", data = by_day, palette = "Greens")
ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 30)
ax1.set(xlabel = "day of the week", ylabel = " Number of serious accidents")
plt.show()

#patterns in time of day/day of week
by_day_hour = accidents.groupby(["day_of_week","real_day","full_hour"])["accident_index"].count().reset_index()
by_day_hour.rename(columns={"day_of_week": "day_of_week", "real_day": "Day of the week","full_hour" : "Time", "accident_index":"accident_index" }, inplace=True)
display(by_day_hour)

heatmap_data = pd.pivot_table(by_day_hour, values='accident_index',
                              index='Time',
                              columns='Day of the week',
                              )
order_d = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data_ord = heatmap_data.reindex(order_d, axis=1)

h = sns.clustermap(heatmap_data_ord,
                    figsize=(10, 8),
                    dendrogram_ratio=(0.1,0.12),
                    row_cluster=False,
                    col_cluster= True,
                    cbar_pos=(0.01, .2, .04, .5),
                    cmap="coolwarm")
h.ax_col_dendrogram.set_title('Time distribution of the car accidents')
h.fig.savefig('destination_path.eps', format='eps', dpi=1000,bbox_inches='tight')

#Let's now explore the data set a bit
accidents.info()
accidents.describe()

accidents.describe().columns



#Lets plot target data
accidents['Serious'].value_counts(sort=False).plot(kind="bar", title = 'Serious')

#new feauture cas_per_veh
accidents['cas_per_veh'] = accidents['number_of_casualties']/accidents['number_of_vehicles']

#separate in numerical and categorical variables
df_num = ['longitude', 'latitude','number_of_vehicles', 'number_of_casualties', 'cas_per_veh']
df_cat = ['accident_severity','day_of_week','first_road_class', 'first_road_number', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'second_road_number', 'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area']

#Let's take a look at the numerical discreete variables distribution
i = 1
sns.set(font_scale = 2)
sns.set_style("white")
sns.set_palette("bright")
plt.figure(figsize=(40, 40))
plt.subplots_adjust(hspace=1)
for feature in df_num:
    plt.subplot(9,4,i)
    sns.histplot(accidents[feature], color='Blue')
    i = i +1

sns.histplot(accidents['cas_per_veh'])

accidents['longitude'] = np.log1p(accidents['longitude'])
accidents['latitude'] = np.log1p(accidents['latitude'])
accidents['number_of_vehicles'] = np.log1p(accidents['number_of_vehicles'])
accidents['number_of_casualties'] = np.log1p(accidents['number_of_casualties'])
accidents['cas_per_veh'] = np.log1p(accidents['cas_per_veh'])

#Let's take a look at the numerical discreete variables distribution after log transform
i = 1
sns.set(font_scale = 2)
sns.set_style("white")
sns.set_palette("bright")
plt.figure(figsize=(40, 40))
plt.subplots_adjust(hspace=1)
for feature in df_num:
    plt.subplot(9,4,i)
    sns.histplot(accidents[feature], color='Blue')
    i = i +1

#correlation of the numerical continues variabled with the target variable
i = 1
sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("Blues_r")
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=1)
for feature in df_num:
    plt.subplot(6,2,i)
    sns.regplot(data=accidents, x=feature,y='Serious')
    i = i +1

#correlation of the numerical continues variabled with the target variable
i = 1
sns.set(font_scale = 1.5)
sns.set_style("white")
sns.set_palette("Blues_r")
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=1)
for feature in df_num:
    plt.subplot(6,2,i)
    sns.regplot(data=accidents, x=feature,y='Serious')
    i = i +1

#Let's take a look at the numeric discreete variables distribution
i = 1
sns.set(font_scale = 2)
sns.set_style("white")
sns.set_palette("bright")
plt.figure(figsize=(40, 40))
plt.subplots_adjust(hspace=1)
for feature in df_cat:
    plt.subplot(9,4,i)
    sns.histplot(accidents[feature], color='Blue')
    i = i +1

accidents['accident_year'].unique()
ready_data = accidents.copy()

ready_data.info()

ready_data['full_hour'] = ready_data['full_hour'].astype('int')

# +
ready_data.drop(['longitude','number_of_casualties','number_of_vehicles','cas_per_veh','accident_index','accident_year','accident_reference','date','time','real_day','first_road_number','second_road_number'], axis = 1, inplace = True)


# -

ready_data.head(10)

#One hot encoding over the data
drop_enc = OneHotEncoder(sparse=False)
transformed = pd.DataFrame(drop_enc.fit_transform(ready_data[['day_of_week','first_road_class', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area']]))
transformed.columns = drop_enc.get_feature_names_out(['day_of_week','first_road_class', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area'])
transformed.index = ready_data.index
ready_data.drop(['day_of_week','first_road_class', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area'], axis=1, inplace = True)
OHE = pd.concat([pd.DataFrame(ready_data),transformed], axis = 1)
"""#Join back to dataframe

ready_data[drop_enc.categories_[0]] = transformed.toarray()


"""

OHE.info()


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


OHE_cleaned = clean_dataset(OHE)

OHE_cleaned.info()

np.any(np.isnan(OHE_cleaned))

np.all(np.isfinite(OHE_cleaned))

#train-test split
X = OHE_cleaned.copy()
y = OHE_cleaned['Serious']
X.drop(['Serious'], axis = 1, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=42, shuffle = True)

X_train.info()

y_train.value_counts()

y_test.value_counts()

y_train.head()

#Synthetic Minority Over-sampling TEchnique (SMOTE) for imbalanced data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 14)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)
X_test_SMOTE, y_test_SMOTE = smote.fit_resample(X_test, y_test)

y_train_SMOTE.value_counts()

# define the scaler
scaler = MinMaxScaler()
# fit on the training dataset
scaler.fit(X_train)
# scale the training dataset
X_train = scaler.transform(X_train)
# scale the test dataset
X_test = scaler.transform(X_test)

# fit on the training dataset approach 2
scaler.fit(X_train_SMOTE)
# scale the training dataset
X_train_SMOTE = scaler.transform(X_train_SMOTE)
# scale the test dataset
X_test_SMOTE = scaler.transform(X_test_SMOTE)


#Using a classification function
def classification(X_tr,y_tr,X_te,y_te,method):    
    #fit and predict
    method.fit(X_tr,y_tr)
    p_train = method.predict(X_tr)
    p_test = method.predict(X_te)
    #because of the imbalance of target data in the beginning lets return the F1 score
    print('-'*20)
    print('train Accuracy')
    print(accuracy_score(y_tr, p_train))
    print('test Accuracy')
    print(accuracy_score(y_te, p_test))
    print('-'*20)
    print('train f1 score')
    print(f1_score(y_tr, p_train))
    print('test f1 score')
    print(f1_score(y_te, p_test))
    print('-'*20)
    print('train ROC AUC score')
    print(roc_auc_score(y_tr, p_train ,average='micro'))
    print('test ROC AUC score')
    print(roc_auc_score(y_te, p_test ,average='micro'))
    print('-'*20)
    print('train Matthews Correlation Coefficient score')
    print(matthews_corrcoef(y_tr, p_train))
    print('test Matthews Correlation Coefficient score')
    print(matthews_corrcoef(y_te, p_test))
    print('-'*20)
    #plotting confusion matrix
    plt.figure(figsize=(15,6))
    ax = sns.heatmap(confusion_matrix(y_te,p_test),cmap='Blues',annot=True)
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    plt.tight_layout()
    plt.show()


#fit the model - LogisticRegression
logreg = LogisticRegression(max_iter = 2000)
classification(X_train,y_train,X_test,y_test,logreg)

#fit the model - LogisticRegression- SMOTE
logreg = LogisticRegression(max_iter = 2000)
classification(X_train_SMOTE,y_train_SMOTE,X_test_SMOTE,y_test_SMOTE,logreg)

#No changes to input trainig data
xgbc=XGBClassifier(random_state=14)
classification(X_train,y_train,X_test,y_test,xgbc)

#SMOTE
xgbc=XGBClassifier(random_state=14)
classification(X_train_SMOTE,y_train_SMOTE,X_test_SMOTE,y_test_SMOTE,xgbc)

#plotting and saving confusion matrix
p_test = xgbc.predict(X_test_SMOTE)
plt.figure(figsize=(10,4))
ax = sns.heatmap(confusion_matrix(y_test_SMOTE,p_test),cmap='Blues',annot=True)
ax.set_title('Confusion matrix')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values')
plt.tight_layout()
plt.show()
figure = ax.get_figure()    
figure.savefig('confusion_matrix.eps', format='eps', dpi=1000,bbox_inches='tight')

#Importance of the features for the logistic regression
lr_fit = logreg.fit(X_train_SMOTE, y_train_SMOTE)
feat_importances_lr = pd.Series(logreg.coef_[0], index=X.columns)
feat_importances_lr.nlargest(10).plot(kind='barh', title = "Logistic Regression")

#Importance of the features for the XGBooster
xgb_fit = xgbc.fit(X_train_SMOTE, y_train_SMOTE)
feat_importances_xgb = pd.Series(xgbc.feature_importances_, index=X.columns)


ax = feat_importances_xgb.nlargest(10).plot(kind='barh', title = "Xtreme Gradient Boosting")
plt.xlabel("Feature importance")
ax.invert_yaxis()
figure = ax.get_figure()    
figure.savefig('FeatImportXGBC.eps', format='eps', dpi=1000,bbox_inches='tight')

#Feature interpretability with SHAP
shap_values = shap.TreeExplainer(xgbc).shap_values(X_train) 

explainer = shap.Explainer(xgbc, X_train)
shap_values = explainer(X_train)

#summary_plot
#shap.plots.beeswarm(shap_values, X_train, X.columns,max_display = 10,show=False)
ax = shap.summary_plot(shap_values,X_train, X.columns,show=False, max_display = 10)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
#figure = ax.get_figure()    
plt.savefig('SHAP_XGBC.eps', format='eps', dpi=1000,bbox_inches='tight')

explainer_SMOTE = shap.Explainer(xgbc, X_train_SMOTE)
shap_values_SMOTE = explainer_SMOTE(X_train_SMOTE)

#summary_plot
#shap.plots.beeswarm(shap_values, X_train, X.columns,max_display = 10,show=False)
shap.summary_plot(shap_values_SMOTE, X_train_SMOTE, X.columns,show=False, max_display = 20)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)

explainer_SMOTE_logreg = shap.Explainer(logreg, X_train_SMOTE)
shap_values_SMOTE_logreg = explainer_SMOTE_logreg(X_train_SMOTE)

shap.summary_plot(shap_values_SMOTE_logreg, X_train_SMOTE, X.columns,show=False, max_display = 10)
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)
