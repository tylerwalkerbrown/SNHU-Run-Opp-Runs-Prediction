```python
import requests
import pandas as pd
from lxml import html
import csv 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
#Scatter matrix
import plotly.express as px
#H Test
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats 
import math
#Decision Tree
from sklearn import tree

#Encoding categorical data 
#from sklearn.preprocession import LabelEncoder, OneHotEncoder
#labellencoder = LabelEncoder()

```


```python
# snhu_22, snhu_21, snhu_19,snhu_18 - all years taken into consideration (hitting)
#snhu_22_p,snhu_21_p, snhu_19_p,snhu_18_p - all years taken into consideration (pitching)
# frames = List used to concat years frames_p = pitchers concat
#df - combination of all years stored in a pandas df
#correlation - corraltion matrix that will be used to analyze r-squared values visually
#sample_mean- sample mean 
#h_mean- pop mean 
#N - sample size 
#dev -standard dev of population
#variables = subset of highly correlated variables
```


```python
#Problems with data 
####PO are innacurate so cannot accurately find the amount of SO per game
####totals at the end of each year need to be dropped or data will be skewed 
####Year 2020 was excluded from analysis (covid-year)
####
```


```python
#Reading in data
snhu_22 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2022', header = 0)
snhu_21 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2021', header = 0)
snhu_19 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2019', header = 0)
snhu_18 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2018', header = 0)
snhu_17 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2017', header = 0)
snhu_16 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2016', header = 0)
snhu_15 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2015', header = 0)
```

# Pitchers Cleaning/Formatting


```python
#Converting to a dataframe instead of a list (hitting)
snhu_15_p = pd.DataFrame(snhu_15[7])
snhu_16_p = pd.DataFrame(snhu_16[7])
snhu_17_p = pd.DataFrame(snhu_17[7])
snhu_18_p = pd.DataFrame(snhu_18[7])
snhu_19_p = pd.DataFrame(snhu_19[7])
snhu_21_p = pd.DataFrame(snhu_21[7])
snhu_22_p = pd.DataFrame(snhu_22[7])

#Converting to a dataframe instead of a list (hitting)
snhu_15 = pd.DataFrame(snhu_15[6])
snhu_16 = pd.DataFrame(snhu_16[6])
snhu_17 = pd.DataFrame(snhu_17[6])
snhu_18 = pd.DataFrame(snhu_18[6])
snhu_19 = pd.DataFrame(snhu_19[6])
snhu_21 = pd.DataFrame(snhu_21[6])
snhu_22 = pd.DataFrame(snhu_22[6])
```


```python
#Setting the indexes to drop the Total Columns 
#Total columns will skew means because of large values
snhu_15_p = snhu_15_p.set_index('Date')
snhu_16_p = snhu_16_p.set_index('Date')
snhu_17_p = snhu_17_p.set_index('Date')
snhu_18_p = snhu_18_p.set_index('Date')
snhu_19_p = snhu_19_p.set_index('Date')
snhu_21_p = snhu_21_p.set_index('Date')
snhu_22_p = snhu_22_p.set_index('Date')

#Setting the indexes to drop the Total Columns 
#Total columns will skew means because of large values
snhu_15 = snhu_15.set_index('Date')
snhu_16 = snhu_16.set_index('Date')
snhu_17 = snhu_17.set_index('Date')
snhu_18 = snhu_18.set_index('Date')
snhu_19 = snhu_19.set_index('Date')
snhu_21 = snhu_21.set_index('Date')
snhu_22 = snhu_22.set_index('Date')
```


```python
#Dropping all year end sum columns 
snhu_15_p = snhu_15_p.drop(index = ('Total'))
snhu_16_p = snhu_16_p.drop(index = ('Total'))
snhu_17_p = snhu_17_p.drop(index = ('Total'))
snhu_18_p = snhu_18_p.drop(index = ('Total'))
snhu_19_p = snhu_19_p.drop(index = ('Total'))
snhu_21_p = snhu_21_p.drop(index = ('Total'))
snhu_22_p = snhu_22_p.drop(index = ('Total'))

#Dropping all year end sum columns 
snhu_15 = snhu_15.drop(index = ('Total'))
snhu_16 = snhu_16.drop(index = ('Total'))
snhu_17 = snhu_17.drop(index = ('Total'))
snhu_18 = snhu_18.drop(index = ('Total'))
snhu_19 = snhu_19.drop(index = ('Total'))
snhu_21 = snhu_21.drop(index = ('Total'))
snhu_22 = snhu_22.drop(index = ('Total'))
```


```python
#Combining all dataframes that we are analyzing 
frames_p = [snhu_15_p,snhu_16_p,snhu_17_p,snhu_18_p,snhu_19_p,snhu_21_p,snhu_22_p]
```


```python
#Concating pitchers data  
frames_p = pd.concat(frames_p)
```


```python
#Dropping repeat columns for the merge
#Data will be captured in the hitting data
frames_p = frames_p.drop(labels = ['Opponent','Score.1','Score','ERA','IP'], axis = 1)

```


```python
#Combining all dataframes that we are analyzing 
frames_h = [snhu_15,snhu_16,snhu_17,snhu_18,snhu_19,snhu_21,snhu_22]
```


```python
#Concating pitchers data  
frames_h = pd.concat(frames_h)
```


```python
#Dropping repeat columns for the merge
#Data will be captured in the hitting data
frames_h = frames_h.drop(labels = ['Loc','Opponent','W/L','R'], axis = 1)

```


```python
frames_h['Score'] = frames_h['Score'].str.split('-', expand = True)[0]
```


```python
#Merging pitcher data and hitting data
df = pd.concat([frames_h,frames_p],axis = 1)
```


```python
#Changing score to int
frames_h['Score'] = frames_h.Score.astype(int)
```

# Anomaly Detection 

# Head and Tail of DF


```python
#Top 5 obervations in df
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>AB</th>
      <th>H</th>
      <th>RBI</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>BB</th>
      <th>IBB</th>
      <th>SB</th>
      <th>...</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>WP</th>
      <th>BK</th>
      <th>HBP</th>
      <th>IBB</th>
      <th>W</th>
      <th>L</th>
      <th>SV</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2/13/2015</th>
      <td>3</td>
      <td>35</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2/14/2015</th>
      <td>3</td>
      <td>35</td>
      <td>8</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2/15/2015</th>
      <td>3</td>
      <td>32</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2/20/2015</th>
      <td>16</td>
      <td>35</td>
      <td>12</td>
      <td>13</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2/20/2015</th>
      <td>6</td>
      <td>36</td>
      <td>9</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
#Bottom 5 observations in df
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>AB</th>
      <th>H</th>
      <th>RBI</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>BB</th>
      <th>IBB</th>
      <th>SB</th>
      <th>...</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>WP</th>
      <th>BK</th>
      <th>HBP</th>
      <th>IBB</th>
      <th>W</th>
      <th>L</th>
      <th>SV</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5/27/2022</th>
      <td>5</td>
      <td>33</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>44</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5/27/2022</th>
      <td>7</td>
      <td>33</td>
      <td>10</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5/29/2022</th>
      <td>7</td>
      <td>35</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6/5/2022</th>
      <td>4</td>
      <td>34</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>46</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6/7/2022</th>
      <td>3</td>
      <td>34</td>
      <td>10</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>46</td>
      <td>12</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>



# EDA 


```python
#Changing the box plot size and layout
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.autolayout"] = True
```


```python
#Looking at the central tendency
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AB</th>
      <th>H</th>
      <th>RBI</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>BB</th>
      <th>IBB</th>
      <th>SB</th>
      <th>CS</th>
      <th>...</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>WP</th>
      <th>BK</th>
      <th>HBP</th>
      <th>IBB</th>
      <th>W</th>
      <th>L</th>
      <th>SV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>...</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
      <td>364.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>34.219780</td>
      <td>9.917582</td>
      <td>6.373626</td>
      <td>1.824176</td>
      <td>0.387363</td>
      <td>0.923077</td>
      <td>4.519231</td>
      <td>0.065934</td>
      <td>2.217033</td>
      <td>0.502747</td>
      <td>...</td>
      <td>1.082418</td>
      <td>0.126374</td>
      <td>0.543956</td>
      <td>0.881868</td>
      <td>0.115385</td>
      <td>0.689560</td>
      <td>0.104396</td>
      <td>21.206044</td>
      <td>5.821429</td>
      <td>0.087912</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.724246</td>
      <td>4.248976</td>
      <td>4.419836</td>
      <td>1.608130</td>
      <td>0.701006</td>
      <td>1.103271</td>
      <td>2.813224</td>
      <td>0.259357</td>
      <td>2.104602</td>
      <td>0.682317</td>
      <td>...</td>
      <td>1.112820</td>
      <td>0.379164</td>
      <td>0.840269</td>
      <td>1.194478</td>
      <td>0.396798</td>
      <td>0.950453</td>
      <td>0.348286</td>
      <td>12.899009</td>
      <td>3.954769</td>
      <td>0.283557</td>
    </tr>
    <tr>
      <th>min</th>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>31.000000</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>34.000000</td>
      <td>10.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>38.000000</td>
      <td>13.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>56.000000</td>
      <td>24.000000</td>
      <td>25.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>18.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>3.000000</td>
      <td>50.000000</td>
      <td>17.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>




```python
#How much Penman outscore oppenant by 
print("On average the Penman outscore their opponent by ", frames_h.Score.mean() - frames_p.R.mean(), "runs")
```

    On average the Penman outscore their opponent by  3.6401098901098896 runs



```python
#Box plot of pitchers data 
df.boxplot()
plt.title('Pitchers Stats')
```




    Text(0.5, 1.0, 'Pitchers Stats')




    
![png](output_25_1.png)
    



```python
#Box plot of pitchers data 
frames_p.boxplot()
plt.title('Pitchers Stats')
```




    Text(0.5, 1.0, 'Pitchers Stats')




    
![png](output_26_1.png)
    



```python
#Box plot of pitchers data 
frames_h.boxplot()
plt.title('Hitter Stats')
```




    Text(0.5, 1.0, 'Hitter Stats')




    
![png](output_27_1.png)
    


# Outlier Removal


```python
#Setting lower and upper bounds for outlier removal 
lower_bound = 0.1
upper_bound = 0.95
```


```python
res = df.AB.quantile([lower_bound,upper_bound])
```


```python
#Pitchers REMOVAL
res = frames_p.H.quantile([lower_bound,upper_bound])
true_index = (res.loc[lower_bound] < frames_p.H.values) & \
        (frames_p.H.values < res.loc[upper_bound])
frames_p = frames_p[true_index]
```


```python
#Pitchers REMOVAL
res = frames_p.BB.quantile([lower_bound,upper_bound])
true_index = (res.loc[lower_bound] < frames_p.BB.values) & \
        (frames_p.BB.values < res.loc[upper_bound])
frames_p = frames_p[true_index]
```


```python
#Pitchers REMOVAL
res = frames_p.WP.quantile([lower_bound,upper_bound])
true_index = (res.loc[lower_bound] < frames_p.WP.values) & \
        (frames_p.WP.values < res.loc[upper_bound])
frames_p = frames_p[true_index]
```


```python
#HITTERS REMOVAL
res = frames_h.BB.quantile([lower_bound,upper_bound])
true_index = (res.loc[lower_bound] < frames_h.BB.values) & \
        (frames_h.BB.values < res.loc[upper_bound])
frames_h = frames_h[true_index]
```


```python
#HITTERS REMOVAL
res = frames_h['3B'].quantile([lower_bound,upper_bound])
true_index = (res.loc[lower_bound] < frames_h['3B'].values) & \
        (frames_h['3B'].values < res.loc[upper_bound])
frames_h = frames_h[true_index]
```

# Continuing Analysis


```python
#Pearson correlation matrix examing r^2
matrix = df.corr(
    method = 'pearson',  # The method of correlation
    min_periods = 1      # Min number of observations required
)
```


```python
#Changing the size of the correlation matrix
sns.set(rc={'figure.figsize':(30,16)})

```


```python
#Correlation Heat Map of hitters
matrix = frames_p.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()
```


    
![png](output_39_0.png)
    



```python
#Correlation Heat Map of hitters
matrix = frames_h.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()
```


    
![png](output_40_0.png)
    



```python
#Correlation Heat Map 
matrix = df.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.show()
```


    
![png](output_41_0.png)
    


# Highly Correlated Variables


```python
#Scatter matrix
pd.plotting.scatter_matrix(frames_h, figsize = (40,40))
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/pandas/plotting/_matplotlib/misc.py:100: UserWarning: Attempting to set identical left == right == 1.0 results in singular transformations; automatically expanding.
      ax.set_xlim(boundaries_list[j])
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/pandas/plotting/_matplotlib/misc.py:101: UserWarning: Attempting to set identical bottom == top == 1.0 results in singular transformations; automatically expanding.
      ax.set_ylim(boundaries_list[i])
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/pandas/plotting/_matplotlib/misc.py:91: UserWarning: Attempting to set identical left == right == 1.0 results in singular transformations; automatically expanding.
      ax.set_xlim(boundaries_list[i])





    array([[<AxesSubplot:xlabel='Score', ylabel='Score'>,
            <AxesSubplot:xlabel='AB', ylabel='Score'>,
            <AxesSubplot:xlabel='H', ylabel='Score'>,
            <AxesSubplot:xlabel='RBI', ylabel='Score'>,
            <AxesSubplot:xlabel='2B', ylabel='Score'>,
            <AxesSubplot:xlabel='3B', ylabel='Score'>,
            <AxesSubplot:xlabel='HR', ylabel='Score'>,
            <AxesSubplot:xlabel='BB', ylabel='Score'>,
            <AxesSubplot:xlabel='IBB', ylabel='Score'>,
            <AxesSubplot:xlabel='SB', ylabel='Score'>,
            <AxesSubplot:xlabel='CS', ylabel='Score'>,
            <AxesSubplot:xlabel='HBP', ylabel='Score'>,
            <AxesSubplot:xlabel='SH', ylabel='Score'>,
            <AxesSubplot:xlabel='SF', ylabel='Score'>,
            <AxesSubplot:xlabel='GDP', ylabel='Score'>,
            <AxesSubplot:xlabel='K', ylabel='Score'>,
            <AxesSubplot:xlabel='PO', ylabel='Score'>,
            <AxesSubplot:xlabel='A', ylabel='Score'>,
            <AxesSubplot:xlabel='E', ylabel='Score'>,
            <AxesSubplot:xlabel='AVG', ylabel='Score'>],
           [<AxesSubplot:xlabel='Score', ylabel='AB'>,
            <AxesSubplot:xlabel='AB', ylabel='AB'>,
            <AxesSubplot:xlabel='H', ylabel='AB'>,
            <AxesSubplot:xlabel='RBI', ylabel='AB'>,
            <AxesSubplot:xlabel='2B', ylabel='AB'>,
            <AxesSubplot:xlabel='3B', ylabel='AB'>,
            <AxesSubplot:xlabel='HR', ylabel='AB'>,
            <AxesSubplot:xlabel='BB', ylabel='AB'>,
            <AxesSubplot:xlabel='IBB', ylabel='AB'>,
            <AxesSubplot:xlabel='SB', ylabel='AB'>,
            <AxesSubplot:xlabel='CS', ylabel='AB'>,
            <AxesSubplot:xlabel='HBP', ylabel='AB'>,
            <AxesSubplot:xlabel='SH', ylabel='AB'>,
            <AxesSubplot:xlabel='SF', ylabel='AB'>,
            <AxesSubplot:xlabel='GDP', ylabel='AB'>,
            <AxesSubplot:xlabel='K', ylabel='AB'>,
            <AxesSubplot:xlabel='PO', ylabel='AB'>,
            <AxesSubplot:xlabel='A', ylabel='AB'>,
            <AxesSubplot:xlabel='E', ylabel='AB'>,
            <AxesSubplot:xlabel='AVG', ylabel='AB'>],
           [<AxesSubplot:xlabel='Score', ylabel='H'>,
            <AxesSubplot:xlabel='AB', ylabel='H'>,
            <AxesSubplot:xlabel='H', ylabel='H'>,
            <AxesSubplot:xlabel='RBI', ylabel='H'>,
            <AxesSubplot:xlabel='2B', ylabel='H'>,
            <AxesSubplot:xlabel='3B', ylabel='H'>,
            <AxesSubplot:xlabel='HR', ylabel='H'>,
            <AxesSubplot:xlabel='BB', ylabel='H'>,
            <AxesSubplot:xlabel='IBB', ylabel='H'>,
            <AxesSubplot:xlabel='SB', ylabel='H'>,
            <AxesSubplot:xlabel='CS', ylabel='H'>,
            <AxesSubplot:xlabel='HBP', ylabel='H'>,
            <AxesSubplot:xlabel='SH', ylabel='H'>,
            <AxesSubplot:xlabel='SF', ylabel='H'>,
            <AxesSubplot:xlabel='GDP', ylabel='H'>,
            <AxesSubplot:xlabel='K', ylabel='H'>,
            <AxesSubplot:xlabel='PO', ylabel='H'>,
            <AxesSubplot:xlabel='A', ylabel='H'>,
            <AxesSubplot:xlabel='E', ylabel='H'>,
            <AxesSubplot:xlabel='AVG', ylabel='H'>],
           [<AxesSubplot:xlabel='Score', ylabel='RBI'>,
            <AxesSubplot:xlabel='AB', ylabel='RBI'>,
            <AxesSubplot:xlabel='H', ylabel='RBI'>,
            <AxesSubplot:xlabel='RBI', ylabel='RBI'>,
            <AxesSubplot:xlabel='2B', ylabel='RBI'>,
            <AxesSubplot:xlabel='3B', ylabel='RBI'>,
            <AxesSubplot:xlabel='HR', ylabel='RBI'>,
            <AxesSubplot:xlabel='BB', ylabel='RBI'>,
            <AxesSubplot:xlabel='IBB', ylabel='RBI'>,
            <AxesSubplot:xlabel='SB', ylabel='RBI'>,
            <AxesSubplot:xlabel='CS', ylabel='RBI'>,
            <AxesSubplot:xlabel='HBP', ylabel='RBI'>,
            <AxesSubplot:xlabel='SH', ylabel='RBI'>,
            <AxesSubplot:xlabel='SF', ylabel='RBI'>,
            <AxesSubplot:xlabel='GDP', ylabel='RBI'>,
            <AxesSubplot:xlabel='K', ylabel='RBI'>,
            <AxesSubplot:xlabel='PO', ylabel='RBI'>,
            <AxesSubplot:xlabel='A', ylabel='RBI'>,
            <AxesSubplot:xlabel='E', ylabel='RBI'>,
            <AxesSubplot:xlabel='AVG', ylabel='RBI'>],
           [<AxesSubplot:xlabel='Score', ylabel='2B'>,
            <AxesSubplot:xlabel='AB', ylabel='2B'>,
            <AxesSubplot:xlabel='H', ylabel='2B'>,
            <AxesSubplot:xlabel='RBI', ylabel='2B'>,
            <AxesSubplot:xlabel='2B', ylabel='2B'>,
            <AxesSubplot:xlabel='3B', ylabel='2B'>,
            <AxesSubplot:xlabel='HR', ylabel='2B'>,
            <AxesSubplot:xlabel='BB', ylabel='2B'>,
            <AxesSubplot:xlabel='IBB', ylabel='2B'>,
            <AxesSubplot:xlabel='SB', ylabel='2B'>,
            <AxesSubplot:xlabel='CS', ylabel='2B'>,
            <AxesSubplot:xlabel='HBP', ylabel='2B'>,
            <AxesSubplot:xlabel='SH', ylabel='2B'>,
            <AxesSubplot:xlabel='SF', ylabel='2B'>,
            <AxesSubplot:xlabel='GDP', ylabel='2B'>,
            <AxesSubplot:xlabel='K', ylabel='2B'>,
            <AxesSubplot:xlabel='PO', ylabel='2B'>,
            <AxesSubplot:xlabel='A', ylabel='2B'>,
            <AxesSubplot:xlabel='E', ylabel='2B'>,
            <AxesSubplot:xlabel='AVG', ylabel='2B'>],
           [<AxesSubplot:xlabel='Score', ylabel='3B'>,
            <AxesSubplot:xlabel='AB', ylabel='3B'>,
            <AxesSubplot:xlabel='H', ylabel='3B'>,
            <AxesSubplot:xlabel='RBI', ylabel='3B'>,
            <AxesSubplot:xlabel='2B', ylabel='3B'>,
            <AxesSubplot:xlabel='3B', ylabel='3B'>,
            <AxesSubplot:xlabel='HR', ylabel='3B'>,
            <AxesSubplot:xlabel='BB', ylabel='3B'>,
            <AxesSubplot:xlabel='IBB', ylabel='3B'>,
            <AxesSubplot:xlabel='SB', ylabel='3B'>,
            <AxesSubplot:xlabel='CS', ylabel='3B'>,
            <AxesSubplot:xlabel='HBP', ylabel='3B'>,
            <AxesSubplot:xlabel='SH', ylabel='3B'>,
            <AxesSubplot:xlabel='SF', ylabel='3B'>,
            <AxesSubplot:xlabel='GDP', ylabel='3B'>,
            <AxesSubplot:xlabel='K', ylabel='3B'>,
            <AxesSubplot:xlabel='PO', ylabel='3B'>,
            <AxesSubplot:xlabel='A', ylabel='3B'>,
            <AxesSubplot:xlabel='E', ylabel='3B'>,
            <AxesSubplot:xlabel='AVG', ylabel='3B'>],
           [<AxesSubplot:xlabel='Score', ylabel='HR'>,
            <AxesSubplot:xlabel='AB', ylabel='HR'>,
            <AxesSubplot:xlabel='H', ylabel='HR'>,
            <AxesSubplot:xlabel='RBI', ylabel='HR'>,
            <AxesSubplot:xlabel='2B', ylabel='HR'>,
            <AxesSubplot:xlabel='3B', ylabel='HR'>,
            <AxesSubplot:xlabel='HR', ylabel='HR'>,
            <AxesSubplot:xlabel='BB', ylabel='HR'>,
            <AxesSubplot:xlabel='IBB', ylabel='HR'>,
            <AxesSubplot:xlabel='SB', ylabel='HR'>,
            <AxesSubplot:xlabel='CS', ylabel='HR'>,
            <AxesSubplot:xlabel='HBP', ylabel='HR'>,
            <AxesSubplot:xlabel='SH', ylabel='HR'>,
            <AxesSubplot:xlabel='SF', ylabel='HR'>,
            <AxesSubplot:xlabel='GDP', ylabel='HR'>,
            <AxesSubplot:xlabel='K', ylabel='HR'>,
            <AxesSubplot:xlabel='PO', ylabel='HR'>,
            <AxesSubplot:xlabel='A', ylabel='HR'>,
            <AxesSubplot:xlabel='E', ylabel='HR'>,
            <AxesSubplot:xlabel='AVG', ylabel='HR'>],
           [<AxesSubplot:xlabel='Score', ylabel='BB'>,
            <AxesSubplot:xlabel='AB', ylabel='BB'>,
            <AxesSubplot:xlabel='H', ylabel='BB'>,
            <AxesSubplot:xlabel='RBI', ylabel='BB'>,
            <AxesSubplot:xlabel='2B', ylabel='BB'>,
            <AxesSubplot:xlabel='3B', ylabel='BB'>,
            <AxesSubplot:xlabel='HR', ylabel='BB'>,
            <AxesSubplot:xlabel='BB', ylabel='BB'>,
            <AxesSubplot:xlabel='IBB', ylabel='BB'>,
            <AxesSubplot:xlabel='SB', ylabel='BB'>,
            <AxesSubplot:xlabel='CS', ylabel='BB'>,
            <AxesSubplot:xlabel='HBP', ylabel='BB'>,
            <AxesSubplot:xlabel='SH', ylabel='BB'>,
            <AxesSubplot:xlabel='SF', ylabel='BB'>,
            <AxesSubplot:xlabel='GDP', ylabel='BB'>,
            <AxesSubplot:xlabel='K', ylabel='BB'>,
            <AxesSubplot:xlabel='PO', ylabel='BB'>,
            <AxesSubplot:xlabel='A', ylabel='BB'>,
            <AxesSubplot:xlabel='E', ylabel='BB'>,
            <AxesSubplot:xlabel='AVG', ylabel='BB'>],
           [<AxesSubplot:xlabel='Score', ylabel='IBB'>,
            <AxesSubplot:xlabel='AB', ylabel='IBB'>,
            <AxesSubplot:xlabel='H', ylabel='IBB'>,
            <AxesSubplot:xlabel='RBI', ylabel='IBB'>,
            <AxesSubplot:xlabel='2B', ylabel='IBB'>,
            <AxesSubplot:xlabel='3B', ylabel='IBB'>,
            <AxesSubplot:xlabel='HR', ylabel='IBB'>,
            <AxesSubplot:xlabel='BB', ylabel='IBB'>,
            <AxesSubplot:xlabel='IBB', ylabel='IBB'>,
            <AxesSubplot:xlabel='SB', ylabel='IBB'>,
            <AxesSubplot:xlabel='CS', ylabel='IBB'>,
            <AxesSubplot:xlabel='HBP', ylabel='IBB'>,
            <AxesSubplot:xlabel='SH', ylabel='IBB'>,
            <AxesSubplot:xlabel='SF', ylabel='IBB'>,
            <AxesSubplot:xlabel='GDP', ylabel='IBB'>,
            <AxesSubplot:xlabel='K', ylabel='IBB'>,
            <AxesSubplot:xlabel='PO', ylabel='IBB'>,
            <AxesSubplot:xlabel='A', ylabel='IBB'>,
            <AxesSubplot:xlabel='E', ylabel='IBB'>,
            <AxesSubplot:xlabel='AVG', ylabel='IBB'>],
           [<AxesSubplot:xlabel='Score', ylabel='SB'>,
            <AxesSubplot:xlabel='AB', ylabel='SB'>,
            <AxesSubplot:xlabel='H', ylabel='SB'>,
            <AxesSubplot:xlabel='RBI', ylabel='SB'>,
            <AxesSubplot:xlabel='2B', ylabel='SB'>,
            <AxesSubplot:xlabel='3B', ylabel='SB'>,
            <AxesSubplot:xlabel='HR', ylabel='SB'>,
            <AxesSubplot:xlabel='BB', ylabel='SB'>,
            <AxesSubplot:xlabel='IBB', ylabel='SB'>,
            <AxesSubplot:xlabel='SB', ylabel='SB'>,
            <AxesSubplot:xlabel='CS', ylabel='SB'>,
            <AxesSubplot:xlabel='HBP', ylabel='SB'>,
            <AxesSubplot:xlabel='SH', ylabel='SB'>,
            <AxesSubplot:xlabel='SF', ylabel='SB'>,
            <AxesSubplot:xlabel='GDP', ylabel='SB'>,
            <AxesSubplot:xlabel='K', ylabel='SB'>,
            <AxesSubplot:xlabel='PO', ylabel='SB'>,
            <AxesSubplot:xlabel='A', ylabel='SB'>,
            <AxesSubplot:xlabel='E', ylabel='SB'>,
            <AxesSubplot:xlabel='AVG', ylabel='SB'>],
           [<AxesSubplot:xlabel='Score', ylabel='CS'>,
            <AxesSubplot:xlabel='AB', ylabel='CS'>,
            <AxesSubplot:xlabel='H', ylabel='CS'>,
            <AxesSubplot:xlabel='RBI', ylabel='CS'>,
            <AxesSubplot:xlabel='2B', ylabel='CS'>,
            <AxesSubplot:xlabel='3B', ylabel='CS'>,
            <AxesSubplot:xlabel='HR', ylabel='CS'>,
            <AxesSubplot:xlabel='BB', ylabel='CS'>,
            <AxesSubplot:xlabel='IBB', ylabel='CS'>,
            <AxesSubplot:xlabel='SB', ylabel='CS'>,
            <AxesSubplot:xlabel='CS', ylabel='CS'>,
            <AxesSubplot:xlabel='HBP', ylabel='CS'>,
            <AxesSubplot:xlabel='SH', ylabel='CS'>,
            <AxesSubplot:xlabel='SF', ylabel='CS'>,
            <AxesSubplot:xlabel='GDP', ylabel='CS'>,
            <AxesSubplot:xlabel='K', ylabel='CS'>,
            <AxesSubplot:xlabel='PO', ylabel='CS'>,
            <AxesSubplot:xlabel='A', ylabel='CS'>,
            <AxesSubplot:xlabel='E', ylabel='CS'>,
            <AxesSubplot:xlabel='AVG', ylabel='CS'>],
           [<AxesSubplot:xlabel='Score', ylabel='HBP'>,
            <AxesSubplot:xlabel='AB', ylabel='HBP'>,
            <AxesSubplot:xlabel='H', ylabel='HBP'>,
            <AxesSubplot:xlabel='RBI', ylabel='HBP'>,
            <AxesSubplot:xlabel='2B', ylabel='HBP'>,
            <AxesSubplot:xlabel='3B', ylabel='HBP'>,
            <AxesSubplot:xlabel='HR', ylabel='HBP'>,
            <AxesSubplot:xlabel='BB', ylabel='HBP'>,
            <AxesSubplot:xlabel='IBB', ylabel='HBP'>,
            <AxesSubplot:xlabel='SB', ylabel='HBP'>,
            <AxesSubplot:xlabel='CS', ylabel='HBP'>,
            <AxesSubplot:xlabel='HBP', ylabel='HBP'>,
            <AxesSubplot:xlabel='SH', ylabel='HBP'>,
            <AxesSubplot:xlabel='SF', ylabel='HBP'>,
            <AxesSubplot:xlabel='GDP', ylabel='HBP'>,
            <AxesSubplot:xlabel='K', ylabel='HBP'>,
            <AxesSubplot:xlabel='PO', ylabel='HBP'>,
            <AxesSubplot:xlabel='A', ylabel='HBP'>,
            <AxesSubplot:xlabel='E', ylabel='HBP'>,
            <AxesSubplot:xlabel='AVG', ylabel='HBP'>],
           [<AxesSubplot:xlabel='Score', ylabel='SH'>,
            <AxesSubplot:xlabel='AB', ylabel='SH'>,
            <AxesSubplot:xlabel='H', ylabel='SH'>,
            <AxesSubplot:xlabel='RBI', ylabel='SH'>,
            <AxesSubplot:xlabel='2B', ylabel='SH'>,
            <AxesSubplot:xlabel='3B', ylabel='SH'>,
            <AxesSubplot:xlabel='HR', ylabel='SH'>,
            <AxesSubplot:xlabel='BB', ylabel='SH'>,
            <AxesSubplot:xlabel='IBB', ylabel='SH'>,
            <AxesSubplot:xlabel='SB', ylabel='SH'>,
            <AxesSubplot:xlabel='CS', ylabel='SH'>,
            <AxesSubplot:xlabel='HBP', ylabel='SH'>,
            <AxesSubplot:xlabel='SH', ylabel='SH'>,
            <AxesSubplot:xlabel='SF', ylabel='SH'>,
            <AxesSubplot:xlabel='GDP', ylabel='SH'>,
            <AxesSubplot:xlabel='K', ylabel='SH'>,
            <AxesSubplot:xlabel='PO', ylabel='SH'>,
            <AxesSubplot:xlabel='A', ylabel='SH'>,
            <AxesSubplot:xlabel='E', ylabel='SH'>,
            <AxesSubplot:xlabel='AVG', ylabel='SH'>],
           [<AxesSubplot:xlabel='Score', ylabel='SF'>,
            <AxesSubplot:xlabel='AB', ylabel='SF'>,
            <AxesSubplot:xlabel='H', ylabel='SF'>,
            <AxesSubplot:xlabel='RBI', ylabel='SF'>,
            <AxesSubplot:xlabel='2B', ylabel='SF'>,
            <AxesSubplot:xlabel='3B', ylabel='SF'>,
            <AxesSubplot:xlabel='HR', ylabel='SF'>,
            <AxesSubplot:xlabel='BB', ylabel='SF'>,
            <AxesSubplot:xlabel='IBB', ylabel='SF'>,
            <AxesSubplot:xlabel='SB', ylabel='SF'>,
            <AxesSubplot:xlabel='CS', ylabel='SF'>,
            <AxesSubplot:xlabel='HBP', ylabel='SF'>,
            <AxesSubplot:xlabel='SH', ylabel='SF'>,
            <AxesSubplot:xlabel='SF', ylabel='SF'>,
            <AxesSubplot:xlabel='GDP', ylabel='SF'>,
            <AxesSubplot:xlabel='K', ylabel='SF'>,
            <AxesSubplot:xlabel='PO', ylabel='SF'>,
            <AxesSubplot:xlabel='A', ylabel='SF'>,
            <AxesSubplot:xlabel='E', ylabel='SF'>,
            <AxesSubplot:xlabel='AVG', ylabel='SF'>],
           [<AxesSubplot:xlabel='Score', ylabel='GDP'>,
            <AxesSubplot:xlabel='AB', ylabel='GDP'>,
            <AxesSubplot:xlabel='H', ylabel='GDP'>,
            <AxesSubplot:xlabel='RBI', ylabel='GDP'>,
            <AxesSubplot:xlabel='2B', ylabel='GDP'>,
            <AxesSubplot:xlabel='3B', ylabel='GDP'>,
            <AxesSubplot:xlabel='HR', ylabel='GDP'>,
            <AxesSubplot:xlabel='BB', ylabel='GDP'>,
            <AxesSubplot:xlabel='IBB', ylabel='GDP'>,
            <AxesSubplot:xlabel='SB', ylabel='GDP'>,
            <AxesSubplot:xlabel='CS', ylabel='GDP'>,
            <AxesSubplot:xlabel='HBP', ylabel='GDP'>,
            <AxesSubplot:xlabel='SH', ylabel='GDP'>,
            <AxesSubplot:xlabel='SF', ylabel='GDP'>,
            <AxesSubplot:xlabel='GDP', ylabel='GDP'>,
            <AxesSubplot:xlabel='K', ylabel='GDP'>,
            <AxesSubplot:xlabel='PO', ylabel='GDP'>,
            <AxesSubplot:xlabel='A', ylabel='GDP'>,
            <AxesSubplot:xlabel='E', ylabel='GDP'>,
            <AxesSubplot:xlabel='AVG', ylabel='GDP'>],
           [<AxesSubplot:xlabel='Score', ylabel='K'>,
            <AxesSubplot:xlabel='AB', ylabel='K'>,
            <AxesSubplot:xlabel='H', ylabel='K'>,
            <AxesSubplot:xlabel='RBI', ylabel='K'>,
            <AxesSubplot:xlabel='2B', ylabel='K'>,
            <AxesSubplot:xlabel='3B', ylabel='K'>,
            <AxesSubplot:xlabel='HR', ylabel='K'>,
            <AxesSubplot:xlabel='BB', ylabel='K'>,
            <AxesSubplot:xlabel='IBB', ylabel='K'>,
            <AxesSubplot:xlabel='SB', ylabel='K'>,
            <AxesSubplot:xlabel='CS', ylabel='K'>,
            <AxesSubplot:xlabel='HBP', ylabel='K'>,
            <AxesSubplot:xlabel='SH', ylabel='K'>,
            <AxesSubplot:xlabel='SF', ylabel='K'>,
            <AxesSubplot:xlabel='GDP', ylabel='K'>,
            <AxesSubplot:xlabel='K', ylabel='K'>,
            <AxesSubplot:xlabel='PO', ylabel='K'>,
            <AxesSubplot:xlabel='A', ylabel='K'>,
            <AxesSubplot:xlabel='E', ylabel='K'>,
            <AxesSubplot:xlabel='AVG', ylabel='K'>],
           [<AxesSubplot:xlabel='Score', ylabel='PO'>,
            <AxesSubplot:xlabel='AB', ylabel='PO'>,
            <AxesSubplot:xlabel='H', ylabel='PO'>,
            <AxesSubplot:xlabel='RBI', ylabel='PO'>,
            <AxesSubplot:xlabel='2B', ylabel='PO'>,
            <AxesSubplot:xlabel='3B', ylabel='PO'>,
            <AxesSubplot:xlabel='HR', ylabel='PO'>,
            <AxesSubplot:xlabel='BB', ylabel='PO'>,
            <AxesSubplot:xlabel='IBB', ylabel='PO'>,
            <AxesSubplot:xlabel='SB', ylabel='PO'>,
            <AxesSubplot:xlabel='CS', ylabel='PO'>,
            <AxesSubplot:xlabel='HBP', ylabel='PO'>,
            <AxesSubplot:xlabel='SH', ylabel='PO'>,
            <AxesSubplot:xlabel='SF', ylabel='PO'>,
            <AxesSubplot:xlabel='GDP', ylabel='PO'>,
            <AxesSubplot:xlabel='K', ylabel='PO'>,
            <AxesSubplot:xlabel='PO', ylabel='PO'>,
            <AxesSubplot:xlabel='A', ylabel='PO'>,
            <AxesSubplot:xlabel='E', ylabel='PO'>,
            <AxesSubplot:xlabel='AVG', ylabel='PO'>],
           [<AxesSubplot:xlabel='Score', ylabel='A'>,
            <AxesSubplot:xlabel='AB', ylabel='A'>,
            <AxesSubplot:xlabel='H', ylabel='A'>,
            <AxesSubplot:xlabel='RBI', ylabel='A'>,
            <AxesSubplot:xlabel='2B', ylabel='A'>,
            <AxesSubplot:xlabel='3B', ylabel='A'>,
            <AxesSubplot:xlabel='HR', ylabel='A'>,
            <AxesSubplot:xlabel='BB', ylabel='A'>,
            <AxesSubplot:xlabel='IBB', ylabel='A'>,
            <AxesSubplot:xlabel='SB', ylabel='A'>,
            <AxesSubplot:xlabel='CS', ylabel='A'>,
            <AxesSubplot:xlabel='HBP', ylabel='A'>,
            <AxesSubplot:xlabel='SH', ylabel='A'>,
            <AxesSubplot:xlabel='SF', ylabel='A'>,
            <AxesSubplot:xlabel='GDP', ylabel='A'>,
            <AxesSubplot:xlabel='K', ylabel='A'>,
            <AxesSubplot:xlabel='PO', ylabel='A'>,
            <AxesSubplot:xlabel='A', ylabel='A'>,
            <AxesSubplot:xlabel='E', ylabel='A'>,
            <AxesSubplot:xlabel='AVG', ylabel='A'>],
           [<AxesSubplot:xlabel='Score', ylabel='E'>,
            <AxesSubplot:xlabel='AB', ylabel='E'>,
            <AxesSubplot:xlabel='H', ylabel='E'>,
            <AxesSubplot:xlabel='RBI', ylabel='E'>,
            <AxesSubplot:xlabel='2B', ylabel='E'>,
            <AxesSubplot:xlabel='3B', ylabel='E'>,
            <AxesSubplot:xlabel='HR', ylabel='E'>,
            <AxesSubplot:xlabel='BB', ylabel='E'>,
            <AxesSubplot:xlabel='IBB', ylabel='E'>,
            <AxesSubplot:xlabel='SB', ylabel='E'>,
            <AxesSubplot:xlabel='CS', ylabel='E'>,
            <AxesSubplot:xlabel='HBP', ylabel='E'>,
            <AxesSubplot:xlabel='SH', ylabel='E'>,
            <AxesSubplot:xlabel='SF', ylabel='E'>,
            <AxesSubplot:xlabel='GDP', ylabel='E'>,
            <AxesSubplot:xlabel='K', ylabel='E'>,
            <AxesSubplot:xlabel='PO', ylabel='E'>,
            <AxesSubplot:xlabel='A', ylabel='E'>,
            <AxesSubplot:xlabel='E', ylabel='E'>,
            <AxesSubplot:xlabel='AVG', ylabel='E'>],
           [<AxesSubplot:xlabel='Score', ylabel='AVG'>,
            <AxesSubplot:xlabel='AB', ylabel='AVG'>,
            <AxesSubplot:xlabel='H', ylabel='AVG'>,
            <AxesSubplot:xlabel='RBI', ylabel='AVG'>,
            <AxesSubplot:xlabel='2B', ylabel='AVG'>,
            <AxesSubplot:xlabel='3B', ylabel='AVG'>,
            <AxesSubplot:xlabel='HR', ylabel='AVG'>,
            <AxesSubplot:xlabel='BB', ylabel='AVG'>,
            <AxesSubplot:xlabel='IBB', ylabel='AVG'>,
            <AxesSubplot:xlabel='SB', ylabel='AVG'>,
            <AxesSubplot:xlabel='CS', ylabel='AVG'>,
            <AxesSubplot:xlabel='HBP', ylabel='AVG'>,
            <AxesSubplot:xlabel='SH', ylabel='AVG'>,
            <AxesSubplot:xlabel='SF', ylabel='AVG'>,
            <AxesSubplot:xlabel='GDP', ylabel='AVG'>,
            <AxesSubplot:xlabel='K', ylabel='AVG'>,
            <AxesSubplot:xlabel='PO', ylabel='AVG'>,
            <AxesSubplot:xlabel='A', ylabel='AVG'>,
            <AxesSubplot:xlabel='E', ylabel='AVG'>,
            <AxesSubplot:xlabel='AVG', ylabel='AVG'>]], dtype=object)




    
![png](output_43_2.png)
    



```python
pd.plotting.scatter_matrix(frames_p, figsize = (40,40))
```




    array([[<AxesSubplot:xlabel='H', ylabel='H'>,
            <AxesSubplot:xlabel='R', ylabel='H'>,
            <AxesSubplot:xlabel='ER', ylabel='H'>,
            <AxesSubplot:xlabel='BB', ylabel='H'>,
            <AxesSubplot:xlabel='SO', ylabel='H'>,
            <AxesSubplot:xlabel='2B', ylabel='H'>,
            <AxesSubplot:xlabel='3B', ylabel='H'>,
            <AxesSubplot:xlabel='HR', ylabel='H'>,
            <AxesSubplot:xlabel='WP', ylabel='H'>,
            <AxesSubplot:xlabel='BK', ylabel='H'>,
            <AxesSubplot:xlabel='HBP', ylabel='H'>,
            <AxesSubplot:xlabel='IBB', ylabel='H'>,
            <AxesSubplot:xlabel='W', ylabel='H'>,
            <AxesSubplot:xlabel='L', ylabel='H'>,
            <AxesSubplot:xlabel='SV', ylabel='H'>],
           [<AxesSubplot:xlabel='H', ylabel='R'>,
            <AxesSubplot:xlabel='R', ylabel='R'>,
            <AxesSubplot:xlabel='ER', ylabel='R'>,
            <AxesSubplot:xlabel='BB', ylabel='R'>,
            <AxesSubplot:xlabel='SO', ylabel='R'>,
            <AxesSubplot:xlabel='2B', ylabel='R'>,
            <AxesSubplot:xlabel='3B', ylabel='R'>,
            <AxesSubplot:xlabel='HR', ylabel='R'>,
            <AxesSubplot:xlabel='WP', ylabel='R'>,
            <AxesSubplot:xlabel='BK', ylabel='R'>,
            <AxesSubplot:xlabel='HBP', ylabel='R'>,
            <AxesSubplot:xlabel='IBB', ylabel='R'>,
            <AxesSubplot:xlabel='W', ylabel='R'>,
            <AxesSubplot:xlabel='L', ylabel='R'>,
            <AxesSubplot:xlabel='SV', ylabel='R'>],
           [<AxesSubplot:xlabel='H', ylabel='ER'>,
            <AxesSubplot:xlabel='R', ylabel='ER'>,
            <AxesSubplot:xlabel='ER', ylabel='ER'>,
            <AxesSubplot:xlabel='BB', ylabel='ER'>,
            <AxesSubplot:xlabel='SO', ylabel='ER'>,
            <AxesSubplot:xlabel='2B', ylabel='ER'>,
            <AxesSubplot:xlabel='3B', ylabel='ER'>,
            <AxesSubplot:xlabel='HR', ylabel='ER'>,
            <AxesSubplot:xlabel='WP', ylabel='ER'>,
            <AxesSubplot:xlabel='BK', ylabel='ER'>,
            <AxesSubplot:xlabel='HBP', ylabel='ER'>,
            <AxesSubplot:xlabel='IBB', ylabel='ER'>,
            <AxesSubplot:xlabel='W', ylabel='ER'>,
            <AxesSubplot:xlabel='L', ylabel='ER'>,
            <AxesSubplot:xlabel='SV', ylabel='ER'>],
           [<AxesSubplot:xlabel='H', ylabel='BB'>,
            <AxesSubplot:xlabel='R', ylabel='BB'>,
            <AxesSubplot:xlabel='ER', ylabel='BB'>,
            <AxesSubplot:xlabel='BB', ylabel='BB'>,
            <AxesSubplot:xlabel='SO', ylabel='BB'>,
            <AxesSubplot:xlabel='2B', ylabel='BB'>,
            <AxesSubplot:xlabel='3B', ylabel='BB'>,
            <AxesSubplot:xlabel='HR', ylabel='BB'>,
            <AxesSubplot:xlabel='WP', ylabel='BB'>,
            <AxesSubplot:xlabel='BK', ylabel='BB'>,
            <AxesSubplot:xlabel='HBP', ylabel='BB'>,
            <AxesSubplot:xlabel='IBB', ylabel='BB'>,
            <AxesSubplot:xlabel='W', ylabel='BB'>,
            <AxesSubplot:xlabel='L', ylabel='BB'>,
            <AxesSubplot:xlabel='SV', ylabel='BB'>],
           [<AxesSubplot:xlabel='H', ylabel='SO'>,
            <AxesSubplot:xlabel='R', ylabel='SO'>,
            <AxesSubplot:xlabel='ER', ylabel='SO'>,
            <AxesSubplot:xlabel='BB', ylabel='SO'>,
            <AxesSubplot:xlabel='SO', ylabel='SO'>,
            <AxesSubplot:xlabel='2B', ylabel='SO'>,
            <AxesSubplot:xlabel='3B', ylabel='SO'>,
            <AxesSubplot:xlabel='HR', ylabel='SO'>,
            <AxesSubplot:xlabel='WP', ylabel='SO'>,
            <AxesSubplot:xlabel='BK', ylabel='SO'>,
            <AxesSubplot:xlabel='HBP', ylabel='SO'>,
            <AxesSubplot:xlabel='IBB', ylabel='SO'>,
            <AxesSubplot:xlabel='W', ylabel='SO'>,
            <AxesSubplot:xlabel='L', ylabel='SO'>,
            <AxesSubplot:xlabel='SV', ylabel='SO'>],
           [<AxesSubplot:xlabel='H', ylabel='2B'>,
            <AxesSubplot:xlabel='R', ylabel='2B'>,
            <AxesSubplot:xlabel='ER', ylabel='2B'>,
            <AxesSubplot:xlabel='BB', ylabel='2B'>,
            <AxesSubplot:xlabel='SO', ylabel='2B'>,
            <AxesSubplot:xlabel='2B', ylabel='2B'>,
            <AxesSubplot:xlabel='3B', ylabel='2B'>,
            <AxesSubplot:xlabel='HR', ylabel='2B'>,
            <AxesSubplot:xlabel='WP', ylabel='2B'>,
            <AxesSubplot:xlabel='BK', ylabel='2B'>,
            <AxesSubplot:xlabel='HBP', ylabel='2B'>,
            <AxesSubplot:xlabel='IBB', ylabel='2B'>,
            <AxesSubplot:xlabel='W', ylabel='2B'>,
            <AxesSubplot:xlabel='L', ylabel='2B'>,
            <AxesSubplot:xlabel='SV', ylabel='2B'>],
           [<AxesSubplot:xlabel='H', ylabel='3B'>,
            <AxesSubplot:xlabel='R', ylabel='3B'>,
            <AxesSubplot:xlabel='ER', ylabel='3B'>,
            <AxesSubplot:xlabel='BB', ylabel='3B'>,
            <AxesSubplot:xlabel='SO', ylabel='3B'>,
            <AxesSubplot:xlabel='2B', ylabel='3B'>,
            <AxesSubplot:xlabel='3B', ylabel='3B'>,
            <AxesSubplot:xlabel='HR', ylabel='3B'>,
            <AxesSubplot:xlabel='WP', ylabel='3B'>,
            <AxesSubplot:xlabel='BK', ylabel='3B'>,
            <AxesSubplot:xlabel='HBP', ylabel='3B'>,
            <AxesSubplot:xlabel='IBB', ylabel='3B'>,
            <AxesSubplot:xlabel='W', ylabel='3B'>,
            <AxesSubplot:xlabel='L', ylabel='3B'>,
            <AxesSubplot:xlabel='SV', ylabel='3B'>],
           [<AxesSubplot:xlabel='H', ylabel='HR'>,
            <AxesSubplot:xlabel='R', ylabel='HR'>,
            <AxesSubplot:xlabel='ER', ylabel='HR'>,
            <AxesSubplot:xlabel='BB', ylabel='HR'>,
            <AxesSubplot:xlabel='SO', ylabel='HR'>,
            <AxesSubplot:xlabel='2B', ylabel='HR'>,
            <AxesSubplot:xlabel='3B', ylabel='HR'>,
            <AxesSubplot:xlabel='HR', ylabel='HR'>,
            <AxesSubplot:xlabel='WP', ylabel='HR'>,
            <AxesSubplot:xlabel='BK', ylabel='HR'>,
            <AxesSubplot:xlabel='HBP', ylabel='HR'>,
            <AxesSubplot:xlabel='IBB', ylabel='HR'>,
            <AxesSubplot:xlabel='W', ylabel='HR'>,
            <AxesSubplot:xlabel='L', ylabel='HR'>,
            <AxesSubplot:xlabel='SV', ylabel='HR'>],
           [<AxesSubplot:xlabel='H', ylabel='WP'>,
            <AxesSubplot:xlabel='R', ylabel='WP'>,
            <AxesSubplot:xlabel='ER', ylabel='WP'>,
            <AxesSubplot:xlabel='BB', ylabel='WP'>,
            <AxesSubplot:xlabel='SO', ylabel='WP'>,
            <AxesSubplot:xlabel='2B', ylabel='WP'>,
            <AxesSubplot:xlabel='3B', ylabel='WP'>,
            <AxesSubplot:xlabel='HR', ylabel='WP'>,
            <AxesSubplot:xlabel='WP', ylabel='WP'>,
            <AxesSubplot:xlabel='BK', ylabel='WP'>,
            <AxesSubplot:xlabel='HBP', ylabel='WP'>,
            <AxesSubplot:xlabel='IBB', ylabel='WP'>,
            <AxesSubplot:xlabel='W', ylabel='WP'>,
            <AxesSubplot:xlabel='L', ylabel='WP'>,
            <AxesSubplot:xlabel='SV', ylabel='WP'>],
           [<AxesSubplot:xlabel='H', ylabel='BK'>,
            <AxesSubplot:xlabel='R', ylabel='BK'>,
            <AxesSubplot:xlabel='ER', ylabel='BK'>,
            <AxesSubplot:xlabel='BB', ylabel='BK'>,
            <AxesSubplot:xlabel='SO', ylabel='BK'>,
            <AxesSubplot:xlabel='2B', ylabel='BK'>,
            <AxesSubplot:xlabel='3B', ylabel='BK'>,
            <AxesSubplot:xlabel='HR', ylabel='BK'>,
            <AxesSubplot:xlabel='WP', ylabel='BK'>,
            <AxesSubplot:xlabel='BK', ylabel='BK'>,
            <AxesSubplot:xlabel='HBP', ylabel='BK'>,
            <AxesSubplot:xlabel='IBB', ylabel='BK'>,
            <AxesSubplot:xlabel='W', ylabel='BK'>,
            <AxesSubplot:xlabel='L', ylabel='BK'>,
            <AxesSubplot:xlabel='SV', ylabel='BK'>],
           [<AxesSubplot:xlabel='H', ylabel='HBP'>,
            <AxesSubplot:xlabel='R', ylabel='HBP'>,
            <AxesSubplot:xlabel='ER', ylabel='HBP'>,
            <AxesSubplot:xlabel='BB', ylabel='HBP'>,
            <AxesSubplot:xlabel='SO', ylabel='HBP'>,
            <AxesSubplot:xlabel='2B', ylabel='HBP'>,
            <AxesSubplot:xlabel='3B', ylabel='HBP'>,
            <AxesSubplot:xlabel='HR', ylabel='HBP'>,
            <AxesSubplot:xlabel='WP', ylabel='HBP'>,
            <AxesSubplot:xlabel='BK', ylabel='HBP'>,
            <AxesSubplot:xlabel='HBP', ylabel='HBP'>,
            <AxesSubplot:xlabel='IBB', ylabel='HBP'>,
            <AxesSubplot:xlabel='W', ylabel='HBP'>,
            <AxesSubplot:xlabel='L', ylabel='HBP'>,
            <AxesSubplot:xlabel='SV', ylabel='HBP'>],
           [<AxesSubplot:xlabel='H', ylabel='IBB'>,
            <AxesSubplot:xlabel='R', ylabel='IBB'>,
            <AxesSubplot:xlabel='ER', ylabel='IBB'>,
            <AxesSubplot:xlabel='BB', ylabel='IBB'>,
            <AxesSubplot:xlabel='SO', ylabel='IBB'>,
            <AxesSubplot:xlabel='2B', ylabel='IBB'>,
            <AxesSubplot:xlabel='3B', ylabel='IBB'>,
            <AxesSubplot:xlabel='HR', ylabel='IBB'>,
            <AxesSubplot:xlabel='WP', ylabel='IBB'>,
            <AxesSubplot:xlabel='BK', ylabel='IBB'>,
            <AxesSubplot:xlabel='HBP', ylabel='IBB'>,
            <AxesSubplot:xlabel='IBB', ylabel='IBB'>,
            <AxesSubplot:xlabel='W', ylabel='IBB'>,
            <AxesSubplot:xlabel='L', ylabel='IBB'>,
            <AxesSubplot:xlabel='SV', ylabel='IBB'>],
           [<AxesSubplot:xlabel='H', ylabel='W'>,
            <AxesSubplot:xlabel='R', ylabel='W'>,
            <AxesSubplot:xlabel='ER', ylabel='W'>,
            <AxesSubplot:xlabel='BB', ylabel='W'>,
            <AxesSubplot:xlabel='SO', ylabel='W'>,
            <AxesSubplot:xlabel='2B', ylabel='W'>,
            <AxesSubplot:xlabel='3B', ylabel='W'>,
            <AxesSubplot:xlabel='HR', ylabel='W'>,
            <AxesSubplot:xlabel='WP', ylabel='W'>,
            <AxesSubplot:xlabel='BK', ylabel='W'>,
            <AxesSubplot:xlabel='HBP', ylabel='W'>,
            <AxesSubplot:xlabel='IBB', ylabel='W'>,
            <AxesSubplot:xlabel='W', ylabel='W'>,
            <AxesSubplot:xlabel='L', ylabel='W'>,
            <AxesSubplot:xlabel='SV', ylabel='W'>],
           [<AxesSubplot:xlabel='H', ylabel='L'>,
            <AxesSubplot:xlabel='R', ylabel='L'>,
            <AxesSubplot:xlabel='ER', ylabel='L'>,
            <AxesSubplot:xlabel='BB', ylabel='L'>,
            <AxesSubplot:xlabel='SO', ylabel='L'>,
            <AxesSubplot:xlabel='2B', ylabel='L'>,
            <AxesSubplot:xlabel='3B', ylabel='L'>,
            <AxesSubplot:xlabel='HR', ylabel='L'>,
            <AxesSubplot:xlabel='WP', ylabel='L'>,
            <AxesSubplot:xlabel='BK', ylabel='L'>,
            <AxesSubplot:xlabel='HBP', ylabel='L'>,
            <AxesSubplot:xlabel='IBB', ylabel='L'>,
            <AxesSubplot:xlabel='W', ylabel='L'>,
            <AxesSubplot:xlabel='L', ylabel='L'>,
            <AxesSubplot:xlabel='SV', ylabel='L'>],
           [<AxesSubplot:xlabel='H', ylabel='SV'>,
            <AxesSubplot:xlabel='R', ylabel='SV'>,
            <AxesSubplot:xlabel='ER', ylabel='SV'>,
            <AxesSubplot:xlabel='BB', ylabel='SV'>,
            <AxesSubplot:xlabel='SO', ylabel='SV'>,
            <AxesSubplot:xlabel='2B', ylabel='SV'>,
            <AxesSubplot:xlabel='3B', ylabel='SV'>,
            <AxesSubplot:xlabel='HR', ylabel='SV'>,
            <AxesSubplot:xlabel='WP', ylabel='SV'>,
            <AxesSubplot:xlabel='BK', ylabel='SV'>,
            <AxesSubplot:xlabel='HBP', ylabel='SV'>,
            <AxesSubplot:xlabel='IBB', ylabel='SV'>,
            <AxesSubplot:xlabel='W', ylabel='SV'>,
            <AxesSubplot:xlabel='L', ylabel='SV'>,
            <AxesSubplot:xlabel='SV', ylabel='SV'>]], dtype=object)




    
![png](output_44_1.png)
    



```python
#Creating a binary numeric target variable for our analysis 
#Function to assign 0 = L and 1 = W
def wins_losses(w_l):
    if w_l == 'W':
        return 1
    else:
        return 0

#Storing the binary variables in result 
df['result'] = df['W/L'].apply(wins_losses)
```


```python
#Storing the variable win/loss into the highly correlated variables dataframe (variables)
frames_p['result'] = df['result']
frames_h['result'] = df.result
```


```python
#Win percentage compared to varibles (hitters)
fig, ax = plt.subplots(figsize=(12,9))
fig.suptitle('Advanced Statistics Correlation with Win Percentage', x = .44, fontsize=20)
win_corr = (frames_h.corr()[['result']].sort_values(by='result',ascending=False))
hm = sns.heatmap(win_corr, annot=True, cmap ='BuGn')
hm.set_yticklabels(hm.get_yticklabels(), fontsize=15)
hm.set_xticklabels(hm.get_xticklabels(), fontsize=15);
```

Runs, RBI, Average are the most correlated variables to winning for SNHU Penman Baseball.


```python
#Win percentage compared to varibles (pitchers)
fig, ax = plt.subplots(figsize=(12,9))
fig.suptitle('Pitching Correlation to Winning', x = .44, fontsize=20)
win_corr = (frames_p.corr().abs()[['result']].sort_values(by='result',ascending=False))
hm = sns.heatmap(win_corr, annot=True, cmap ='YlGn')
hm.set_yticklabels(hm.get_yticklabels(), fontsize=15)
hm.set_xticklabels(hm.get_xticklabels(), fontsize=15);
```

Runs, ERA, ER are the most correlated variables to winning for SNHU Penman Baseball.


```python
#Creating a function to look at home,away and neutral
def home_away(h_a):   
    if h_a == 'vs':
        return 'home'
    elif h_a == 'at':
        return 'away'
    else:
        return 'neutral'
df['site'] = df['Loc'].apply(home_away)
```


```python
era_avg = pd.DataFrame(df.groupby(['site'])['ER'].mean())
#Bar plot looking at average ERA at different sites
plt.bar(era_avg.index,era_avg.ER, color ='gold',
        width = 0.4)
plt.xticks(ha='right', rotation=55, fontsize=35, fontname='monospace')
plt.yticks(rotation=55, fontsize=35, fontname='monospace')
plt.xlabel("Location",fontsize=35)
plt.ylabel("Average ERA",fontsize=35)
plt.title("Average ERA (H,A,N)",fontsize=35)
plt.show()
```

Looking at the average era of each of the different locations. This holds the common understanding of home field advantage showing a lower era while the Penman play at home. Neutral settings are actually holding the lowest era out of the three options. The reason that neutral is an option is becuase in college baseball you have times that fields may flood, snow (turf complex) and then playoffs are played at neutral settings if the host team is kicked out of the tournament.


```python
ba_avg = pd.DataFrame(df.groupby(['site'])['AVG'].mean())
#Bar plot looking at average runs at different sites
plt.bar(ba_avg.index,ba_avg.AVG, color ='blue',
        width = 0.4)
plt.xticks(ha='right', rotation=55, fontsize=35, fontname='monospace')
plt.yticks(rotation=55, fontsize=35, fontname='monospace')
plt.xlabel("Location",fontsize=35)
plt.ylabel("Average BA",fontsize=35)
plt.title("Average BA (H,A,N)",fontsize=35)
plt.show()
```

Home and away does not sem to have a huge impact for SNHU hitters. Neutral settings are higher on average compared to both home and away.

# Exporting CSV to Avoid Hardcoding 


```python
import os
```


```python
os.chdir("Desktop/Portfolio Project")
```


```python
#df = pd.read_csv("snhu_data.csv")
```


```python
df.to_csv("snhu_data.csv")
```

#                                      Statistical Modeling


```python
#Changing the box plot size and layout
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.autolayout"] = True
```


```python
#Packages needed for random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
```


```python
#Subsetting all the variables used in random forest
#selection = df[['oH','oR','ER','ERA','opps','penman','AVG','AB','R','H','RBI','result']]
```


```python
#Variables that will be taken into the random forest model 
#selection.info()
```

1. Call the model

2. Fit to train dataset

3. Use fitted model

4. Evaluate model

# Packages for Modeling


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
```

# Modeling Hitters (SNHU runs target)


```python
#Selecting everyhting except target
x_hitters = frames_h.loc[:, frames_h.columns != 'Score']

#Grabbing the target variables
#y = variables.iloc[:, -1]
y_hitters = frames_h.Score
```


```python
#Splitting the sets up for train and test
#Splitting test size 25% of all observations
x_train,x_test,y_train,y_test = train_test_split(x_hitters,y_hitters, test_size = 0.25, random_state = 42)
```


```python
y_test
```




    Date
    5/27/2018     3
    2/28/2016    10
    3/15/2015     5
    4/21/2015    10
    5/20/2016    11
    03/31/18     18
    5/1/2021      6
    3/16/2015    12
    3/11/2022    20
    04/26/15      7
    4/17/2016     9
    4/27/2022    10
    2/22/2019     7
    2/21/2015    16
    5/17/2019     8
    4/9/2021      7
    3/15/2019    17
    Name: Score, dtype: int64




```python
#Using standard scaler to improve the model performance
ss = StandardScaler()

X_sc_train = ss.fit_transform(x_train)
X_sc_test = ss.fit_transform(x_test)
```

# Testing Models Hitter Statistics


```python
#LogisticRegression
lr = LogisticRegression()

pd.DataFrame(pd.DataFrame(cross_validate(lr, X_sc_train, y_train, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.012093</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000272</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.116364</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.970488</td>
    </tr>
  </tbody>
</table>
</div>




```python
#RandomForest
rfc = RandomForestClassifier(n_estimators=100)

pd.DataFrame(pd.DataFrame(cross_validate(rfc, X_sc_train, y_train, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.065954</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.004400</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.254545</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#GuassianNB
gnb = GaussianNB()

pd.DataFrame(pd.DataFrame(cross_validate(gnb, X_sc_train, y_train, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.000980</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000526</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.120000</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.902073</td>
    </tr>
  </tbody>
</table>
</div>




```python
#cross_validate
svc = SVC(kernel='linear')

pd.DataFrame(pd.DataFrame(cross_validate(svc, X_sc_train, y_train, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.002257</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000442</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.156364</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#GradientBoostingClassifier
gbc = GradientBoostingClassifier()
pd.DataFrame(pd.DataFrame(cross_validate(gbc, X_sc_train, y_train, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.451764</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000633</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.216364</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Fitting Hitters (opponent runs)

Logistic regression returned the best accuracy so we will fit the model to logistic regression.


```python
#Assigning the classifier to rf variable 
RF = RandomForestClassifier()
RF.fit(x_train,y_train)
```




    RandomForestClassifier()




```python
#creating predictor set 
y_pred = RF.predict(x_test)
```


```python
#Accuracy score
accuracy_score(y_test,y_pred)
```




    0.17647058823529413



# Subsetting Target and Variables For Pitchers


```python
#Selecting everyhting except target
x_pitchers = frames_p.loc[:, ~frames_p.columns.isin(['R','vs','W/L', 'Loc','result','ER'])]

#Grabbing the target variables
#y = variables.iloc[:, -1]
y_pitchers = frames_p.R
```


```python
#Splitting the sets up for train and test
#Splitting test size 25% of all observations
x_train_p,x_test_p,y_train_p,y_test_p = train_test_split(x_pitchers,y_pitchers, test_size = 0.15, random_state = 42)
```


```python
#Using standard scaler to improve the model performance
ss = StandardScaler()

X_sc_train_p = ss.fit_transform(x_train_p)
X_sc_test_p = ss.fit_transform(x_test_p)
```

# Testing Pitcher Models


```python
#Logistic Regression
pd.DataFrame(pd.DataFrame(cross_validate(lr, X_sc_train_p, y_train_p, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.011020</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000263</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.172549</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.600414</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Random forest
pd.DataFrame(pd.DataFrame(cross_validate(rfc, X_sc_train_p, y_train_p, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.067478</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.004438</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.239869</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#GuassianNB
pd.DataFrame(pd.DataFrame(cross_validate(gnb, X_sc_train_p, y_train_p, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.001697</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000699</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.172549</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.424886</td>
    </tr>
  </tbody>
</table>
</div>




```python
#cross_validate
pd.DataFrame(pd.DataFrame(cross_validate(svc, X_sc_train_p, y_train_p, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.002799</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000528</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.193464</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.767039</td>
    </tr>
  </tbody>
</table>
</div>




```python
#GradientBoostingClassifier
pd.DataFrame(pd.DataFrame(cross_validate(gbc, X_sc_train_p, y_train_p, return_train_score=True)).mean())
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/model_selection/_split.py:676: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5.
      warnings.warn(





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.306386</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000661</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.172549</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Fit/ Accuracy (Pitchers Runs)


```python
#Assigning the classifier to rf variable 
lr = LogisticRegression()
lr.fit(x_train_p,y_train_p)
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    LogisticRegression()




```python
#creating predictor set 
y_pred_p = lr.predict(x_test_p)
```


```python
#Accuracy score
accuracy_score(y_test_p,y_pred_p)
```




    0.1875



# Assessment


```python
from sklearn import metrics
```


```python
#Accuracy score for random forest model 
accuracy_score(y_test, y_pred)
```




    0.17647058823529413




```python
#Confusion Matrix 
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
conf_matrix
```




    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])




```python
#Plotting confusion matrix 
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6))
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
```

    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/mlxtend/plotting/plot_confusion_matrix.py:102: RuntimeWarning: invalid value encountered in true_divide
      normed_conf_mat = conf_mat.astype("float") / total_samples



    
![png](output_106_1.png)
    



```python
#Classification report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               3       0.00      0.00      0.00         1
               5       0.00      0.00      0.00         1
               6       0.00      0.00      0.00         1
               7       0.00      0.00      0.00         3
               8       0.00      0.00      0.00         1
               9       0.00      0.00      0.00         1
              10       0.50      1.00      0.67         3
              11       0.00      0.00      0.00         1
              12       0.00      0.00      0.00         1
              14       0.00      0.00      0.00         0
              16       0.00      0.00      0.00         1
              17       0.00      0.00      0.00         1
              18       0.00      0.00      0.00         1
              20       0.00      0.00      0.00         1
    
        accuracy                           0.18        17
       macro avg       0.04      0.07      0.05        17
    weighted avg       0.09      0.18      0.12        17
    


    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /Users/tylerbrown/opt/anaconda3/lib/python3.9/site-packages/sklearn/metrics/_classification.py:1318: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



```python
#Mean squared error 
mean_squared_error(y_test,y_pred)
```




    6.352941176470588




```python
#Root mean squared error
np.sqrt(mean_squared_error(y_test,y_pred))
```




    2.5205041512504174



# Results vs Actual (SNHU Runs)


```python
comparison = pd.DataFrame(y_test.astype(float))
```


```python
comparison['predict'] = y_pred
comparison.index = pd.to_datetime(comparison.index)
```


```python
#Comparing the results of the predicted vs the actual
comparison.sort_index().cumsum().plot(linewidth=4.0)
plt.xlabel('Predictions', fontsize=30)
plt.ylabel('Actuals', fontsize=30)
plt.xticks(ha='right', rotation=55, fontsize=35, fontname='monospace')
plt.yticks(rotation=55, fontsize=35, fontname='monospace')
plt.title('Predicted vs Actual Logistic Regression (SNHU RUNS)', fontsize=30)
plt.legend(loc=2,prop={'size': 30})
plt.show()
```


    
![png](output_113_0.png)
    


# Results vs Actual (Runs let in)


```python
comparison_p = pd.DataFrame(y_test_p.astype(float))
```


```python
comparison_p['predict'] = y_pred_p
comparison_p.index = pd.to_datetime(comparison_p.index)
```


```python
#Comparing the results of the predicted vs the actual
comparison_p.sort_index().cumsum().plot(linewidth=4.0)
plt.xlabel('Predictions', fontsize=30)
plt.ylabel('Actuals', fontsize=30)
plt.xticks(ha='right', rotation=55, fontsize=35, fontname='monospace')
plt.yticks(rotation=55, fontsize=35, fontname='monospace')
plt.title('Predicted vs Actual Logistic Regression (SNHU RUNS)', fontsize=30)
plt.legend(loc=2,prop={'size': 30})
plt.show()
```


    
![png](output_117_0.png)
    


# Comparing Results Accuracy


```python
#Runs scored for snhu
print("The gradient boost predicted at" ,y_test_p.sum()/y_pred_p.sum() *100, "% accuracy for SNHU runs scored.")

```

    The gradient boost predicted at 91.80327868852459 % accuracy for SNHU runs scored.



```python
#Accuracy of runs scored agianst snhu
print("The gradient boost predicted at" ,y_test.sum()/y_pred.sum() *100, "% accuracy for runs scored against SNHU.")

```

    The gradient boost predicted at 98.87640449438202 % accuracy for runs scored against SNHU.

