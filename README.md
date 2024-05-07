# Bank_loans_Analysis-using-Machine-learning-Algorithms
# <p align="center"> Bank Loan Defaulter prediction  </p>
# <p align="center">![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/a6b29630-7554-4f77-b3a7-97a4f8d16599)

</p>

### Introduction

Loan default prediction is a crucial task for banks and financial institutions to mitigate risks associated with lending.This project utilizes machine learning algorithms to analyze 
customer data and predict whether a customer is likely to default on their loan.


**Tools:-** 
- python3
- pandas
- numpy
- sk learn
- seaborn
- matplotlib
- jupyter lab
- Excel

[Datasets Used](https://docs.google.com/spreadsheets/d/1Yp_rcOS2TbVn-wHUIsCeCzkeDP7MIPLP/edit?usp=sharing&ouid=102868121048017441192&rtpof=true&sd=true )

[Python Script (Code)](cyber_security.ipynb)

[Ppt presentation](sql_prjct.pptx)

### Features 

- Data preprocessing: Clean and prepare the transactional data for analysis.
  
- Supervised learning: Train classification models to classify transactions as fraudulent or legitimate.
  
- Model evaluation: Assess the performance of the models using relevant metrics such as precision, recall, and F1-score.


## Requirements

- Python 3

- Libraries: NumPy, pandas, Sklearn, etc.

- Jupyter Lab


### Information about our dataset

```py
data_ex.info()
```

###### result
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/b4cf3029-dad6-4481-8382-4c68f3057372)

 ##### We are not having any null values.

 ```py
#2.missing values treatment
data_ex.isnull().sum()
```

###### result
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/2da543ca-1bb1-4661-8e1d-05378d9ba063)


- ### Outlier treatment 

##### We have used box plot for Outlier detection.

```py
#3.outliar treatment 

for column in data_ex:
    plt.figure(figsize=(10,1))
    sns.boxplot(x=data_ex[column])
    plt.title(f"Boxplot of {column}")
    plt.show() 
```
###### Result 

![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/68f746dc-f173-4ad2-b2b1-cc308abe1ff2)
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/70dd2ea1-63eb-4e8f-bbba-d75bdab7e714)
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/9c0ed21f-7d30-47c8-ac6c-34de19918e89)


- #### checking the distribution of the data

```py
#4.checking the distribution of the data
#Using random distribution 
for col in d1.columns:
    sns.displot(d1[col],kde = True)
```
###### Result 
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/66945841-6c0b-488e-b3c5-8ae2d7824790)
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/e1bf106e-a36d-4e2c-bfff-56b04fbb8076)

![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/978d4625-8d1b-4835-a241-686d81acd8b9)
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/397641c6-3ba8-48cd-91cf-390a16fa0a44)

![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/a925c88e-348d-49d8-be09-1b21aff34492)
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/1b0ff932-5df6-45d7-91b3-5024c61d17fc)

![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/28c51d0e-8feb-4bc2-b07f-db956461d578)
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/713b22b9-ae6c-4dd0-bbe7-79fe93e471e7)


- #### Correlation 
```py
#6.correlation analysis 
d1.corr()
sns.heatmap(d1.corr().abs(),annot=True)
plt.show()
```
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/f0e93bdd-9856-4ed5-8ff3-90e0ce05a15f)

- ### Model Evaluation
 - #### Decision Tree Classifier Model
```py
ds=DecisionTreeClassifier(max_depth=3)
ds.fit(x_train,y_train)
train_pred=ds.predict(x_train)
test_pred=ds.predict(x_test)
print(accuracy_score(train_pred,y_train))
print(accuracy_score(test_pred,y_test))

import matplotlib.pyplot as plt
from  sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(ds,feature_names=x.columns.tolist(),class_names=["0","1"],filled=True)
plt.show()
```

###### Result 
![image](https://github.com/surajbisht06/Bank_loans_Analysis-using-Machine-learning-Algorithms/assets/158066824/928f7208-821a-48cb-af66-2c0c33b2333f)

- ### Conclusion

- We are having 9 variables in our dataset. We look for some null values and duplicate values but there is no null values and duplicates. After we detect the outliers using box and most of the column having huge number of outliers so we created 3 copies of our original data d1,d2,d3 and then we apply outlier treatment in all the all the valiables in d1. Then we have apply outlier trreatment in some variabls where evere having less ammount of variabels.Then we apply all teh Maching learning models.


- we have applied so many machine learning models to get the perfect result without ofverfitting, and finally we get the final model using Logistic regression model there we got 99% accuracy and there is no overfitting.























