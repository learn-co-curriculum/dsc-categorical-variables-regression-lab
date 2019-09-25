
# Dealing with Categorical Variables - Lab

## Introduction
In this lab, you'll explore the Boston Housing dataset for categorical variables, and you'll transform your data so you'll be able to use categorical data as predictors!

## Objectives
You will be able to:
* Identify and inspect the categorical variables in the Boston housing dataset 
* Categorize inputs that aren't categorical 
* Create new datasets with dummy variables  

## Importing the Boston Housing dataset

Let's start by importing the Boston Housing dataset. This dataset is available in Scikit-Learn, and can be imported running the cell below: 


```python
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
```


```python
# __SOLUTION__ 
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
```

If you'll inspect `boston` now, you'll see that this basically returns a dictionary. Let's have a look at what exactly is stored in the dictionary by looking at the dictionary keys: 


```python
# Print boston
```


```python
# Look at the keys
```


```python
# __SOLUTION__ 
print(boston)
```


```python
# __SOLUTION__ 
boston.keys()
```

Let's create a Pandas DataFrame with the data (which are the features, **not including the target**) and the feature names as column names.


```python
boston_features = None
```


```python
# __SOLUTION__ 
boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
```

Now look at the first five rows of `boston_features`:  


```python
# Inspect the first few rows
```


```python
# __SOLUTION__ 
boston_features.head()
```

For your reference, we copied the attribute information below. Additional information can be found here: http://scikit-learn.org/stable/datasets/index.html#boston-dataset
- CRIM: per capita crime rate by town
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX: nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population

Recall that the values corresponding to the data key are the features. The target is not included. For this dataset, the target is the median value of owner-occupied homes in $1000s and the values can be accessed using the target key. Using the target key, convert the target to a separate DataFrame and set `'MEDV'` as the column name.


```python
boston_target = None

# Inspect the first few rows

```


```python
# __SOLUTION__ 
boston_target = pd.DataFrame(boston.target, columns = ["MEDV"])
boston_target.head()
```

The target is described as: 
- MEDV: Median value of owner-occupied homes in $1000s

Next, let's merge the target and the predictors in one DataFrame `boston_df`: 


```python
boston_df = None
boston_df.head()
```


```python
# __SOLUTION__ 
boston_df = pd.concat([boston_target, boston_features], axis=1)
boston_df.head()
```

Let's inspect these 13 features using `.describe()` and `.info()`


```python
# Use .describe()
```


```python
# __SOLUTION__ 
boston_features.describe()
```


```python
# Use .info()
```


```python
# __SOLUTION__ 
boston_features.info()
```

Now, take a look at the scatter plots for each predictor with the target on the y-axis.


```python
import pandas as pd
import matplotlib.pyplot as plt

# Create scatter plots

```


```python
# __SOLUTION__ 
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(boston_features)[0:4], axes):
    boston_df.plot(kind='scatter', x= xcol, y="MEDV", ax=ax, alpha=0.4, color='b')
```


```python
# __SOLUTION__ 
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(list(boston_features)[4:8], axes):
    boston_df.plot(kind='scatter', x= xcol, y="MEDV", ax=ax, alpha=0.4, color='b')
```


```python
# __SOLUTION__ 
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16,3))

for xcol, ax in zip(list(boston_features)[5:19], axes):
    boston_df.plot(kind='scatter', x= xcol, y="MEDV", ax=ax, alpha=0.4, color='b')
```

## To categorical: binning

If you created your scatterplots correctly, you'll notice that except for `CHAS` (the Charles River Dummy variable), there is clearly no categorical data. You will have seen though that `RAD` and `TAX` have more of a vertical-looking structure like the one seen in the lesson, and that there is less of a "cloud"-looking structure compared to most other variables. It is difficult to justify a linear pattern between predictor and target here. In this situation, it might make sense to restructure data into bins so that they're treated as categorical variables. We'll start by showing how this can be done for `RAD` and then it's your turn to do this for `TAX`.

### RAD

Look at the structure of `RAD` to decide how to create your bins. 


```python
boston_df['RAD'].describe()
```


```python
# __SOLUTION__ 
boston_df["RAD"].describe()
```


```python
# First, create bins based on the values observed. 5 values will result in 4 bins
bins = [0, 3, 4 , 5, 24]

# Use pd.cut()
bins_rad = pd.cut(boston_df['RAD'], bins)
```


```python
# __SOLUTION__ 
# First, create bins based on the values observed. 5 values will result in 4 bins
bins = [0, 3, 4 , 5, 24]

# Use pd.cut()
bins_rad = pd.cut(boston_df['RAD'], bins)
```


```python
# Using pd.cut() returns unordered categories. Transform this to ordered categories 
bins_rad = bins_rad.cat.as_ordered()
bins_rad.head()
```


```python
# __SOLUTION__ 
# Using pd.cut() returns unordered categories. Transform this to ordered categories 
bins_rad = bins_rad.cat.as_ordered()
bins_rad.head()
```


```python
# Inspect the result
bins_rad.value_counts().plot(kind='bar')
```


```python
# __SOLUTION__ 
# Inspect the result
bins_rad.value_counts().plot(kind='bar')
```


```python
# Replace the existing 'RAD' column
boston_df["RAD"] = bins_rad
```


```python
# __SOLUTION__ 
# Replace the existing 'RAD' column
boston_df["RAD"]=bins_rad
```

### TAX

Split the `TAX` column up in 5 categories. You can chose the bins as desired but make sure they're pretty well-balanced.


```python
# Repeat everything for "TAX"
```


```python
# __SOLUTION__ 
boston_df["TAX"].describe()
```


```python
# __SOLUTION__ 
# First, create bins for based on the values observed. 5 values will result in 4 bins
bins = [0, 250, 300, 360, 460, 712]
# Use pd.cut()
bins_tax = pd.cut(boston_df['TAX'], bins)
# Using pd.cut() returns unordered categories. Transform this to ordered categories 
bins_tax = bins_tax.cat.as_ordered()
bins_tax.head()
```


```python
# __SOLUTION__ 
# Check if the result is balanced
bins_tax.value_counts().plot(kind='bar')
```


```python
# __SOLUTION__ 
boston_df["TAX"] = bins_tax
```

## Perform label encoding 


```python
# Perform label encoding and replace in boston_df

```


```python
# Inspect first few columns
```


```python
# __SOLUTION__ 
boston_df["RAD"] = boston_df["RAD"].cat.codes
boston_df["TAX"] = boston_df["TAX"].cat.codes
```


```python
# __SOLUTION__ 
boston_df.head()
```

## Create dummy variables

Create dummy variables, and make sure their column names contain `'TAX'` and `'RAD'` remembering to drop the first. Add the new dummy variables to `boston_df` and remove the old `'TAX'` and `'RAD'` columns.


```python
# Create dummpy variables for TAX and RAD columns

```


```python
# __SOLUTION__ 
tax_dummy = pd.get_dummies(bins_tax, prefix="TAX", drop_first=True)
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD", drop_first=True)
```


```python
# __SOLUTION__ 
boston_df = boston_df.drop(["RAD","TAX"], axis=1)
```


```python
# __SOLUTION__ 
boston_df.head()
```


```python
# __SOLUTION__ 
boston_df = pd.concat([boston_df, rad_dummy, tax_dummy], axis=1)
boston_df.head()
```

Note how you end up with 19 columns now!

## Summary

In this lab, you practiced your knowledge of categorical variables on the Boston Housing dataset!
