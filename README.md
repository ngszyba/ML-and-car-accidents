# [Jakub Szybinski](https://www.linkedin.com/in/jakubszybinski/)
## Data Science portfolio
### Project 1: Factors impacting serious car accidents
***
* I grouped serious accidents (>=3 casualities) according to the time of the day and day of the week to estimate possible trends. We can clearly see that rush hours during the working days are hot spots for the serious car accidents.
<img src="/Figures/Heatmap_days.png" width="600">

* Having separated the dataset into train and test data and corrected for the target variable imbalance, I trained Logistic Regression and Xtreme Gradient Boosting models on the train set. I have evaluated performance of the models on the test data.  
*  
| **Model** | **Accuracy** | **F1-score** | **Matthews Correlation Coefficient** |
| :----: | :----:| :----:| :----:|
| Logistic Regression     | 0.6304 |  0.6304 |  0.2609 |            
| Xtreme Gradient Boosting | 0.9650 |  0.9637 |  0.9321 |                     
* The Precision and Recall were illustrated further with help of a Confusion matrix (for the Xtreme Gradient Boosting).
![](/Figures/confusion_matrix.png)
* I took a closer look into what data features are important for the model
![](/Figures/FeatImportXGBC.png)
* To reconfirm the important features and also get additional info on direction in which it shifts the model interpretation I took advantage of SHAP plots.
![](/Figures/SHAP_XGBC_2.png)
* This way I have picked the features of the data set most important for classification into a "Serious accident" category. We can draw first conclusions that areas with certain speed limits, facilities nearby pedestrian road-crossing points, specific road types, as well as current road surface / weather conditions should draw more attention from authorities to help to combat high numbers of Serious road accidents. This is an example of how Machine Learning can indirectly help improving safety and life quality of citizens.
   
   
***


