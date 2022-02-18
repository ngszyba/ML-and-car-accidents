# [Jakub Szybinski](https://www.linkedin.com/in/jakubszybinski/)
## Data Science portfolio
### Project 1: Factors impacting serious car accidents
***
* I grouped serious accidents (>=3 casualities) according to the time of the day and day of the week to estimate possible trends.
![](/Figures/Heatmap_days.png)

* Having separated the dataset into train and test data and corrected for the target variable imbalance, I trained Logistic Regression and Xtreme Gradient Boosting models on the train set. I have evaluated performance of the models on the test data.  
*  
| **Model** | **Accuracy** | **F1-score** | **Matthews Correlation Coefficient** |
| :----: | :----:| :----:| :----:|
| Logistic Regression     | 0.6304 |  0.6304 |  0.2609 |            
| Xtreme Gradient Boosting | 0.9650 |  0.9637 |  0.9321 |                     
* The Precision and Recall were illustrated further with help of a Confusion matrix (for the Xtreme Gradient Boosting).
![](/Figures/confusion_matrix.png)
* For each of the models I looked into what data categories are important for the model
![](/Figures/Xgb_features.png)
* This way I have picked the most important categorical variables of the data set and performed exploratory data analysis of those to learn which characteristics stand out in serious accidents compared with other accidents (plotted as % of Serious accidents in all accident types). For example: Serious accidents tend to happen more often in areas with a higher speed limit. In addition, serious accidents occur more often at dual carriageway (road type - 3) than at a roundabouts (road type 1).

![](/Figures/speed.png)
![](/Figures/road_type.png)
   
   
***


