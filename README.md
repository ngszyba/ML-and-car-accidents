# Data Science portfolio
### Project 1: Factors impacting serious car accidents
* I grouped serious accidents (>=3 casualities) according to the time of the day and day of the week to estimate possible trends.
![](/Figures/heatmap.png)


***
* Having separated the dataset into train and test data, I trained Logistic Regression, Decision Tree, and Xtreme Gradient Boosting models on the train set. I have evaluated performance of the 3 models by cross validation
| Model	                  | Baseline Performance  | 
| ------------------------|-----------------------| 
| Logistic Regression     | 0.9469                | 
| Xtreme Gradient Boosting| 0.9467                | 
| Decision Tree           | 0.9443                |   
* For each of the models I looked into what data categories are important for the model
![](/Figures/Xgb_features.png)
* I have picked the most important categorical variables of the data set and performed exploratory data analysis to learn which characteristics stand out in serious accidents compared with other accidents.

![](/Figures/speed.png)
![](/Figures/road_type.png)


