import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
accidents_raw = pd.read_csv(r'./data/accident-data.csv')
accidents_raw.head()
lookup = pd.read_csv(r'./data/road-safety-lookups.csv')
lookup.head()

# Create a report that covers the following:
#
# 1. What time of day and day of the week do most serious accidents happen?
# 2. Are there any patterns in the time of day/ day of the week when serious accidents occur?
# 3. What characteristics stand out in serious accidents compared with other accidents?
# 4. On what areas would you recommend the planning team focus their brainstorming efforts to reduce serious accidents?


#serious accidents vs other accidents
accidents = accidents_raw.copy()
accidents["full_hour"] = [x[:2] for x in accidents["time"].astype('str')]
days = {1:"Sunday", 2:"Monday", 3:"Tuesday", 4:"Wednesday", 5:"Thursday", 6:"Friday", 7:"Saturday"}
accidents["real_day"] = [days[x] for x in accidents["day_of_week"]]
serious = accidents[accidents["number_of_casualties"] >= 3]
other = accidents[accidents["accident_severity"] < 3]
serious.head(10)

#What time of the day most accidents happen
by_time = serious.groupby("full_hour")["accident_index"].count().reset_index()
ax = sns.barplot(x = "full_hour", y = "accident_index", data = by_time, palette = "Blues")
ax.set(xlabel = "hour of the day", ylabel = " Number of serious accidents")
plt.show()

#What day of the week most accidents happen
by_day_1 = serious.groupby(["day_of_week","real_day"])["accident_index"].count().reset_index()
by_day = by_day_1.reindex([1,2,3,4,5,6,7,0])
ax1 = sns.barplot(x = "real_day", y = "accident_index", data = by_day, palette = "Greens")
ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 30)
ax1.set(xlabel = "day of the week", ylabel = " Number of serious accidents")
plt.show()

#patterns in time of day/day of week
by_day_hour = serious.groupby(["day_of_week","real_day","full_hour"])["accident_index"].count().reset_index()
display(by_day_hour)

heatmap_data = pd.pivot_table(by_day_hour, values='accident_index',
                              index='full_hour',
                              columns='real_day',
                              )
order_d = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heatmap_data_ord = heatmap_data.reindex(order_d, axis=1)

h = sns.clustermap(heatmap_data_ord,
                    figsize=(10, 7),
                    row_cluster=False,
                    col_cluster= False,
                    cbar_pos=(0.1, .2, .03, .4),
                    cmap="coolwarm")

#Let's now explore the data set a bit
accidents.info()
accidents.describe()

accidents.describe().columns

#add serious column giving 1 for serious accident with 3 or more casualities and 0 for other types
def what_accident(row):
    if row['number_of_casualties'] >= 3:
        return 1
    else:
        return 0
accidents['Serious'] = accidents.apply(what_accident, axis=1)
display(accidents.tail(20))

#separate in numerical and categorical variables
df_num = accidents[['longitude', 'latitude','number_of_vehicles', 'number_of_casualties']]
df_cat = accidents[['Serious','accident_severity','day_of_week','first_road_class', 'first_road_number', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'second_road_number', 'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area']]

#check distribution of the numerical data
for i in df_num.columns:
    plt.hist(df_num[i], bins = 20)
    plt.title(i)
    plt.show()
#if we want to go ahead with using number of vehicles or casualities we would need to normalize them first
#check correlations
sns.heatmap(df_num.corr())
print()
#is there a trend towards serious accidents?
pd.pivot_table(accidents, index = 'Serious', values = ['longitude', 'latitude','number_of_vehicles', 'number_of_casualties'])

#get dummies for the categories
all_dummies = pd.get_dummies(accidents[['Serious','day_of_week','first_road_class', 'first_road_number', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'second_road_number', 'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area']])

#import neccessary ML modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#split data into train and test

train, test = train_test_split(all_dummies, test_size = 0.15)

features = ['day_of_week','first_road_class', 'first_road_number', 'road_type', 'speed_limit',
       'junction_detail', 'junction_control', 'second_road_class',
       'second_road_number', 'pedestrian_crossing_human_control',
       'pedestrian_crossing_physical_facilities', 'light_conditions',
       'weather_conditions', 'road_surface_conditions',
       'special_conditions_at_site', 'carriageway_hazards',
       'urban_or_rural_area']
X_train = train[features]
y_train = train['Serious']

X_test = test[features]
y_test = test['Serious']

#logistic regression
lr = LogisticRegression(max_iter = 2000)
cv_lr = cross_val_score(lr,X_train,y_train,cv=5)
print(cv_lr)
print(cv_lr.mean())

#Decision Tree
dt = DecisionTreeClassifier(min_samples_split = 100)
cv_dt = cross_val_score(dt,X_train,y_train,cv=5)
print(cv_dt)
print(cv_dt.mean())

#XGBClassifier
from xgboost import XGBClassifier
xgb = XGBClassifier(random_state =1)
cv_xgb = cross_val_score(xgb,X_train,y_train,cv=5)
print(cv_xgb)
print(cv_xgb.mean())

#Model Building (Baseline Validation Performance)
Model = ["Logistic Regression", "Decision Tree", "Xtreme Gradient Boosting"]
Baseline_performance = [cv_lr.mean(), cv_dt.mean(), cv_xgb.mean()]
model_summary = pd.DataFrame(list(zip(Model, Baseline_performance)), columns = ['Model', 'Baseline Performance']).sort_values('Baseline Performance', axis = 0, ascending = False)
display(model_summary)

#Importance of the features for the logistic regression
lr_fit = lr.fit(X_train, y_train)
feat_importances_lr = pd.Series(lr.coef_[0], index=X_train.columns)
feat_importances_lr.nlargest(20).plot(kind='barh', title = "Logistic Regression")

#Importance of the features for the Decision Tree Classifier
dt_fit = dt.fit(X_train, y_train)
feat_importances_dt = pd.Series(dt.feature_importances_, index=X_train.columns)
feat_importances_dt.nlargest(20).plot(kind='barh', title = "Decision Tree")

#Importance of the features for the XGBooster
xgb_fit = xgb.fit(X_train, y_train)
feat_importances_xgb = pd.Series(xgb.feature_importances_, index=X_train.columns)
feat_importances_xgb.nlargest(20).plot(kind='barh', title = "Xtreme Gradient Boosting")

#Interestingly some important features that influence ML models strongly are:
#speed_limit
#road_type
#first_road_number
#light_conditions

#Let's explore them independently
a = pd.pivot_table(accidents, columns = 'Serious', index = 'speed_limit',values = 'accident_index', aggfunc = 'count')
a["% of Other"] = a[0]/(a[0]+a[1])*100
a["% of Serious"] = a[1]/(a[0]+a[1])*100
a.plot(y = ["% of Serious"],kind = 'bar')
#plt.axhline(y=2,linewidth=1, color='r',linestyle = '--' )

b = pd.pivot_table(accidents, columns = 'Serious', index = 'road_type',values = 'accident_index', aggfunc = 'count' )
b["% of Other"] = b[0]/(b[0]+b[1])*100
b["% of Serious"] = b[1]/(b[0]+b[1])*100
b.plot(y = ["% of Serious"],kind = 'bar')
#plt.axhline(y=3,linewidth=1, color='r',linestyle = '--' )

c = pd.pivot_table(accidents, columns = 'Serious', index = 'first_road_number',values = 'accident_index', aggfunc = 'count' )
c["% of Other"] = c[0]/(c[0]+c[1])*100
c["% of Serious"] = c[1]/(c[0]+c[1])*100
c_high = c["% of Serious"][c["% of Serious"] > 50]
c_high.plot(y = ["% of Serious"],kind = 'bar')
#plt.axhline(y=15,linewidth=1, color='r',linestyle = '--' )

d = pd.pivot_table(accidents, columns = 'Serious', index = 'light_conditions',values = 'accident_index', aggfunc = 'count' )
d["% of Other"] = d[0]/(d[0]+d[1])*100
d["% of Serious"] = d[1]/(d[0]+d[1])*100
d.plot(y = ["% of Serious"],kind = 'bar')
#plt.axhline(y=15,linewidth=1, color='r',linestyle = '--' )
