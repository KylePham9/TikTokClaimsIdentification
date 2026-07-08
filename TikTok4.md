# **TikTok Project**
**Course 4 - Regression Analysis: Simplify complex data relationships**


```python
import numpy as np
import pandas as pd
import platform
import statsmodels
print('Python version: ', platform.python_version())
print('numpy version: ', np.__version__)
print('pandas version: ', pd.__version__)
print('statsmodels version: ', statsmodels.__version__)
```

    Python version:  3.13.5
    numpy version:  2.1.3
    pandas version:  2.2.3
    statsmodels version:  0.14.4



```python
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Import packages for data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample

# Import packages for data modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

```


```python
data = pd.read_csv("tiktok_dataset.csv")
```


```python
data.head(5)
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
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get number of rows and columns
data.shape
```




    (19382, 12)




```python
# Get data types of columns
data.dtypes
```




    #                             int64
    claim_status                 object
    video_id                      int64
    video_duration_sec            int64
    video_transcription_text     object
    verified_status              object
    author_ban_status            object
    video_view_count            float64
    video_like_count            float64
    video_share_count           float64
    video_download_count        float64
    video_comment_count         float64
    dtype: object




```python
# Get basic information
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19382 entries, 0 to 19381
    Data columns (total 12 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   #                         19382 non-null  int64  
     1   claim_status              19084 non-null  object 
     2   video_id                  19382 non-null  int64  
     3   video_duration_sec        19382 non-null  int64  
     4   video_transcription_text  19084 non-null  object 
     5   verified_status           19382 non-null  object 
     6   author_ban_status         19382 non-null  object 
     7   video_view_count          19084 non-null  float64
     8   video_like_count          19084 non-null  float64
     9   video_share_count         19084 non-null  float64
     10  video_download_count      19084 non-null  float64
     11  video_comment_count       19084 non-null  float64
    dtypes: float64(5), int64(3), object(4)
    memory usage: 1.8+ MB



```python
# Generate basic descriptive stats
data.describe()
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
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19382.000000</td>
      <td>1.938200e+04</td>
      <td>19382.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
      <td>19084.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9691.500000</td>
      <td>5.627454e+09</td>
      <td>32.421732</td>
      <td>254708.558688</td>
      <td>84304.636030</td>
      <td>16735.248323</td>
      <td>1049.429627</td>
      <td>349.312146</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5595.245794</td>
      <td>2.536440e+09</td>
      <td>16.229967</td>
      <td>322893.280814</td>
      <td>133420.546814</td>
      <td>32036.174350</td>
      <td>2004.299894</td>
      <td>799.638865</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.234959e+09</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4846.250000</td>
      <td>3.430417e+09</td>
      <td>18.000000</td>
      <td>4942.500000</td>
      <td>810.750000</td>
      <td>115.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9691.500000</td>
      <td>5.618664e+09</td>
      <td>32.000000</td>
      <td>9954.500000</td>
      <td>3403.500000</td>
      <td>717.000000</td>
      <td>46.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14536.750000</td>
      <td>7.843960e+09</td>
      <td>47.000000</td>
      <td>504327.000000</td>
      <td>125020.000000</td>
      <td>18222.000000</td>
      <td>1156.250000</td>
      <td>292.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>19382.000000</td>
      <td>9.999873e+09</td>
      <td>60.000000</td>
      <td>999817.000000</td>
      <td>657830.000000</td>
      <td>256130.000000</td>
      <td>14994.000000</td>
      <td>9599.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for missing values
data.isna().sum()
```




    #                             0
    claim_status                298
    video_id                      0
    video_duration_sec            0
    video_transcription_text    298
    verified_status               0
    author_ban_status             0
    video_view_count            298
    video_like_count            298
    video_share_count           298
    video_download_count        298
    video_comment_count         298
    dtype: int64




```python
# Drop rows with missing values
data = data.dropna(axis = 0)
```


```python
# Display first few rows after handling missing values
data.head()
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
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for duplicates
data.isna().sum()
```




    #                           0
    claim_status                0
    video_id                    0
    video_duration_sec          0
    video_transcription_text    0
    verified_status             0
    author_ban_status           0
    video_view_count            0
    video_like_count            0
    video_share_count           0
    video_download_count        0
    video_comment_count         0
    dtype: int64



Check for and handle outliers.


```python
# Create a boxplot to visualize distribution of `video_duration_sec`
fig = sns.boxplot(x = data['video_duration_sec'])
fig.set_title('Video Duration Seconds')
fig

```




    <Axes: title={'center': 'Video Duration Seconds'}, xlabel='video_duration_sec'>




    
![png](output_14_1.png)
    



```python
# Create a boxplot to visualize distribution of `video_view_count`
fig = sns.boxplot(x = data['video_view_count'])
fig.set_title('Video View Count')
fig;
```


    
![png](output_15_0.png)
    



```python
# Create a boxplot to visualize distribution of `video_like_count`
fig = sns.boxplot(x = data['video_like_count'])
fig.set_title('Video Like Count')
fig;
```


    
![png](output_16_0.png)
    



```python
# Create a boxplot to visualize distribution of `video_comment_count`
fig = sns.boxplot(x = data['video_comment_count'])
fig.set_title('Video Comment Count')
fig;
```


    
![png](output_17_0.png)
    



```python
# Check for and handle outliers for video_like_count
percentile25 = data['video_like_count'].quantile(0.25)
percentile75 = data['video_like_count'].quantile(0.75)

iqr = percentile75-percentile25
upperlimit = percentile75 + 1.5 * iqr

data.loc[data["video_like_count"] > upperlimit, "video_like_count"] = upperlimit
```

Check class balance of the target variable. Remember, the goal is to predict whether the user of a given post is verified or unverified.


```python
# Check class balance
data['verified_status'].value_counts(normalize = True)
```




    verified_status
    not verified    0.93712
    verified        0.06288
    Name: proportion, dtype: float64



Approximately 94.2% of the dataset represents videos posted by unverified accounts and 5.8% represents videos posted by verified accounts. So the outcome variable is not very balanced.

Use resampling to create class balance in the outcome variable, if needed.


```python
# Identify data points from majority and minority classes
data_majority = data[data["verified_status"] == "not verified"]
data_minority = data[data["verified_status"] == "verified"]
# Upsample the minority class (which is "verified")
data_minority_upsampled = resample(data_minority,
                                 replace=True,                 # to sample with replacement
                                 n_samples=len(data_majority), # to match majority class
                                 random_state=0)               # to create reproducible results

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled]).reset_index(drop=True)
# Display new class counts
data_upsampled["verified_status"].value_counts()
```




    verified_status
    not verified    17884
    verified        17884
    Name: count, dtype: int64




```python
# Get the average `video_transcription_text` length for claims and the average `video_transcription_text` length for opinions
data_upsampled[["verified_status", "video_transcription_text"]].groupby(by="verified_status")[["video_transcription_text"]].agg(func=lambda array: np.mean([len(text) for text in array]))
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
      <th>video_transcription_text</th>
    </tr>
    <tr>
      <th>verified_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not verified</th>
      <td>89.401141</td>
    </tr>
    <tr>
      <th>verified</th>
      <td>84.569559</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data_upsampled["text_length"] = data_upsampled["video_transcription_text"].apply(func=lambda text: len(text))
```


```python
data_upsampled.head()
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
      <th>#</th>
      <th>claim_status</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>verified_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>claim</td>
      <td>7017666017</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>claim</td>
      <td>4014381136</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>not verified</td>
      <td>active</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>claim</td>
      <td>9859838091</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>not verified</td>
      <td>active</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>claim</td>
      <td>1866847991</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>not verified</td>
      <td>active</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>claim</td>
      <td>7105231098</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>not verified</td>
      <td>active</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
    </tr>
  </tbody>
</table>
</div>



Visualize the distribution of `video_transcription_text` length for videos posted by verified accounts and videos posted by unverified accounts.


```python
# Visualize the distribution of `video_transcription_text` length for videos posted by verified accounts and videos posted by unverified accounts
sns.histplot(data=data_upsampled, stat="count", multiple="stack", x="text_length", kde=False, palette="pastel", 
             hue="verified_status", element="bars", legend=True)
plt.title("Seaborn Stacked Histogram")
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for videos posted by verified accounts and videos posted by unverified accounts")
plt.show()
```


    
![png](output_28_0.png)
    



```python
# Code a correlation matrix to help determine most correlated variables
data_upsampled.corr(numeric_only=True)
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
      <th>#</th>
      <th>video_id</th>
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>#</th>
      <td>1.000000</td>
      <td>-0.000853</td>
      <td>-0.011729</td>
      <td>-0.697007</td>
      <td>-0.626385</td>
      <td>-0.504015</td>
      <td>-0.487096</td>
      <td>-0.413799</td>
      <td>-0.193677</td>
    </tr>
    <tr>
      <th>video_id</th>
      <td>-0.000853</td>
      <td>1.000000</td>
      <td>0.011859</td>
      <td>0.002554</td>
      <td>0.005993</td>
      <td>0.010515</td>
      <td>0.008753</td>
      <td>0.013983</td>
      <td>-0.007083</td>
    </tr>
    <tr>
      <th>video_duration_sec</th>
      <td>-0.011729</td>
      <td>0.011859</td>
      <td>1.000000</td>
      <td>0.013589</td>
      <td>0.004494</td>
      <td>0.002206</td>
      <td>0.003989</td>
      <td>-0.004586</td>
      <td>-0.002981</td>
    </tr>
    <tr>
      <th>video_view_count</th>
      <td>-0.697007</td>
      <td>0.002554</td>
      <td>0.013589</td>
      <td>1.000000</td>
      <td>0.856937</td>
      <td>0.711313</td>
      <td>0.690048</td>
      <td>0.583485</td>
      <td>0.244693</td>
    </tr>
    <tr>
      <th>video_like_count</th>
      <td>-0.626385</td>
      <td>0.005993</td>
      <td>0.004494</td>
      <td>0.856937</td>
      <td>1.000000</td>
      <td>0.832146</td>
      <td>0.805543</td>
      <td>0.686647</td>
      <td>0.216693</td>
    </tr>
    <tr>
      <th>video_share_count</th>
      <td>-0.504015</td>
      <td>0.010515</td>
      <td>0.002206</td>
      <td>0.711313</td>
      <td>0.832146</td>
      <td>1.000000</td>
      <td>0.710117</td>
      <td>0.620182</td>
      <td>0.171651</td>
    </tr>
    <tr>
      <th>video_download_count</th>
      <td>-0.487096</td>
      <td>0.008753</td>
      <td>0.003989</td>
      <td>0.690048</td>
      <td>0.805543</td>
      <td>0.710117</td>
      <td>1.000000</td>
      <td>0.857679</td>
      <td>0.173396</td>
    </tr>
    <tr>
      <th>video_comment_count</th>
      <td>-0.413799</td>
      <td>0.013983</td>
      <td>-0.004586</td>
      <td>0.583485</td>
      <td>0.686647</td>
      <td>0.620182</td>
      <td>0.857679</td>
      <td>1.000000</td>
      <td>0.149750</td>
    </tr>
    <tr>
      <th>text_length</th>
      <td>-0.193677</td>
      <td>-0.007083</td>
      <td>-0.002981</td>
      <td>0.244693</td>
      <td>0.216693</td>
      <td>0.171651</td>
      <td>0.173396</td>
      <td>0.149750</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a heatmap to visualize how correlated variables are
sns.heatmap(
    data_upsampled[["video_duration_sec", "claim_status", "author_ban_status", "video_view_count", 
                    "video_like_count", "video_share_count", "video_download_count", "video_comment_count", "text_length"]]
    .corr(numeric_only=True), 
    annot=True, 
    cmap="crest")
plt.title("Heatmap of the dataset")
plt.show()
```


    
![png](output_30_0.png)
    



```python
# Select outcome variable
y = data_upsampled['verified_status']
```


```python
# Select features
X = data_upsampled[["video_duration_sec", "claim_status", "author_ban_status", "video_view_count", "video_share_count", "video_download_count", "video_comment_count"]]
X.head()
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
      <th>video_duration_sec</th>
      <th>claim_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>claim</td>
      <td>under review</td>
      <td>343296.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>claim</td>
      <td>active</td>
      <td>140877.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>claim</td>
      <td>active</td>
      <td>902185.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>claim</td>
      <td>active</td>
      <td>437506.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>claim</td>
      <td>active</td>
      <td>56167.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```


```python
# Get shape of each training and testing set
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((26826, 7), (8942, 7), (26826,), (8942,))




```python
# Check data types
X_train.dtypes
```




    video_duration_sec        int64
    claim_status             object
    author_ban_status        object
    video_view_count        float64
    video_share_count       float64
    video_download_count    float64
    video_comment_count     float64
    dtype: object




```python
# Get unique values in `claim_status`
X_train['claim_status'].unique()

```




    array(['opinion', 'claim'], dtype=object)




```python
# Get unique values in `author_ban_status`
X_train['author_ban_status'].unique()

```




    array(['active', 'under review', 'banned'], dtype=object)




```python
# Select the training features that needs to be encoded
X_train_to_encode = X_train[["claim_status", "author_ban_status"]]
X_train_to_encode.head()
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
      <th>claim_status</th>
      <th>author_ban_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33058</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>20491</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>25583</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>18474</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>27312</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Set up an encoder for one-hot encoding the categorical features
X_encoder = OneHotEncoder(drop = 'first', sparse_output = False)

```


```python
# Fit and transform the training features using the encoder
X_train_encoded = X_encoder.fit_transform(X_train_to_encode)
```


```python
# Get feature names from encoder
X_encoder.get_feature_names_out()
```




    array(['claim_status_opinion', 'author_ban_status_banned',
           'author_ban_status_under review'], dtype=object)




```python
# Display first few rows of encoded training features
X_train_encoded
```




    array([[1., 0., 0.],
           [1., 0., 0.],
           [1., 0., 0.],
           ...,
           [1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])




```python
# Place encoded training features (which is currently an array) into a dataframe
X_train_encoded_df = pd.DataFrame(data=X_train_encoded, columns=X_encoder.get_feature_names_out())
X_train_encoded_df.head()

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
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display first few rows of `X_train` with `claim_status` and `author_ban_status` columns dropped (since these features are being transformed to numeric)
X_train_encoded_df = pd.DataFrame(data=X_train_encoded, columns=X_encoder.get_feature_names_out())

```


```python
# Concatenate `X_train` and `X_train_encoded_df` to form the final dataframe for training data (`X_train_final`)
X_train_final = pd.concat([X_train.drop(columns=["claim_status", "author_ban_status"]).reset_index(drop=True), X_train_encoded_df], axis=1)
X_train_final.head()
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
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>2252.0</td>
      <td>23.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52</td>
      <td>6664.0</td>
      <td>550.0</td>
      <td>53.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>6327.0</td>
      <td>257.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>57</td>
      <td>1702.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>3842.0</td>
      <td>101.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check data type of outcome variable
y_train.dtypes
```




    dtype('O')




```python
# Get unique values of outcome variable
y_train.unique()
```




    array(['verified', 'not verified'], dtype=object)




```python
# Set up an encoder for one-hot encoding the categorical outcome variable
y_encoder = OneHotEncoder(drop='first', sparse_output=False)
```


```python
# Encode the training outcome variable
y_train_final = y_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_train_final
```




    array([1., 1., 1., ..., 1., 1., 0.])




```python
# Construct a logistic regression model and fit it to the training set
log_clf = LogisticRegression(random_state=0, max_iter=800).fit(X_train_final, y_train_final)
```


```python
# Select the testing features that needs to be encoded
X_test_to_encode = X_test[["claim_status", "author_ban_status"]]
X_test_to_encode.head()
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
      <th>claim_status</th>
      <th>author_ban_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21061</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>31748</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
    <tr>
      <th>20197</th>
      <td>claim</td>
      <td>active</td>
    </tr>
    <tr>
      <th>5727</th>
      <td>claim</td>
      <td>active</td>
    </tr>
    <tr>
      <th>11607</th>
      <td>opinion</td>
      <td>active</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Transform the testing features using the encoder
X_test_encoded = X_encoder.transform(X_test_to_encode)
X_test_encoded
```




    array([[1., 0., 0.],
           [1., 0., 0.],
           [0., 0., 0.],
           ...,
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.]])




```python
# Place encoded testing features (which is currently an array) into a dataframe
X_test_encoded_df = pd.DataFrame(data=X_test_encoded, columns=X_encoder.get_feature_names_out())
X_test_encoded_df.head()

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
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Display first few rows of `X_test` with `claim_status` and `author_ban_status` columns dropped (since these features are being transformed to numeric)
X_test.drop(columns=["claim_status", "author_ban_status"])
X_test.head()
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
      <th>video_duration_sec</th>
      <th>claim_status</th>
      <th>author_ban_status</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21061</th>
      <td>41</td>
      <td>opinion</td>
      <td>active</td>
      <td>2118.0</td>
      <td>57.0</td>
      <td>5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>31748</th>
      <td>27</td>
      <td>opinion</td>
      <td>active</td>
      <td>5701.0</td>
      <td>157.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20197</th>
      <td>31</td>
      <td>claim</td>
      <td>active</td>
      <td>449767.0</td>
      <td>75385.0</td>
      <td>5956.0</td>
      <td>1789.0</td>
    </tr>
    <tr>
      <th>5727</th>
      <td>19</td>
      <td>claim</td>
      <td>active</td>
      <td>792813.0</td>
      <td>56597.0</td>
      <td>5146.0</td>
      <td>3413.0</td>
    </tr>
    <tr>
      <th>11607</th>
      <td>54</td>
      <td>opinion</td>
      <td>active</td>
      <td>2044.0</td>
      <td>68.0</td>
      <td>19.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Concatenate `X_test` and `X_test_encoded_df` to form the final dataframe for training data (`X_test_final`)
X_test_final = pd.concat([X_test.drop(columns=["claim_status", "author_ban_status"]).reset_index(drop=True), X_test_encoded_df], axis=1)
X_test_final.head()

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
      <th>video_duration_sec</th>
      <th>video_view_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>claim_status_opinion</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>2118.0</td>
      <td>57.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27</td>
      <td>5701.0</td>
      <td>157.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>449767.0</td>
      <td>75385.0</td>
      <td>5956.0</td>
      <td>1789.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>792813.0</td>
      <td>56597.0</td>
      <td>5146.0</td>
      <td>3413.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54</td>
      <td>2044.0</td>
      <td>68.0</td>
      <td>19.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Test the logistic regression model. Use the model to make predictions on the encoded testing set.


```python
# Use the logistic regression model to get predictions on the encoded testing set
y_pred = log_clf.predict(X_test_final)
```

Display the predictions on the encoded testing set.


```python
# Display the predictions on the encoded testing set
y_pred
```




    array([1., 1., 0., ..., 1., 0., 1.])



Display the true labels of the testing set.


```python
# Display the true labels of the testing set
y_test
```




    21061        verified
    31748        verified
    20197        verified
    5727     not verified
    11607    not verified
                 ...     
    14756    not verified
    26564        verified
    14800    not verified
    35705        verified
    31060        verified
    Name: verified_status, Length: 8942, dtype: object




```python
# Encode the testing outcome variable
y_test_final = y_encoder.transform(y_test.values.reshape(-1, 1)).ravel()
y_test_final
```




    array([1., 1., 1., ..., 0., 1., 1.])




```python
# Get shape of each training and testing set
X_train_final.shape, y_train_final.shape, X_test_final.shape, y_test_final.shape
```




    ((26826, 8), (26826,), (8942, 8), (8942,))




```python
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test_final, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.show()
```


    
![png](output_64_0.png)
    



```python
# Create a classification report
(3758+2044) / (3758 + 725 + 2044 + 2415)
target_labels = ["verified", "not verified"]
print(classification_report(y_test_final, y_pred, target_names=target_labels))
```

                  precision    recall  f1-score   support
    
        verified       0.74      0.52      0.61      4459
    not verified       0.63      0.82      0.71      4483
    
        accuracy                           0.67      8942
       macro avg       0.69      0.67      0.66      8942
    weighted avg       0.69      0.67      0.66      8942
    



```python
# Get the feature names from the model and the model coefficients (which represent log-odds ratios)
# Place into a DataFrame for readability
pd.DataFrame(data={"Feature Name":log_clf.feature_names_in_, "Model Coefficient":log_clf.coef_[0]})
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
      <th>Feature Name</th>
      <th>Model Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>video_duration_sec</td>
      <td>-2.415405e-03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>video_view_count</td>
      <td>-8.060024e-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>video_share_count</td>
      <td>4.721134e-06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>video_download_count</td>
      <td>-1.737111e-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>video_comment_count</td>
      <td>3.297259e-04</td>
    </tr>
    <tr>
      <th>5</th>
      <td>claim_status_opinion</td>
      <td>1.635816e+00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>author_ban_status_banned</td>
      <td>-4.399011e-01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>author_ban_status_under review</td>
      <td>-1.097845e-01</td>
    </tr>
  </tbody>
</table>
</div>


