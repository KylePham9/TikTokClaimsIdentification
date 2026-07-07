# **TikTok Project**
**Course 5 - The Nuts and bolts of machine learning**


```python
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


```


```python
data = pd.read_csv("tiktok_dataset.csv")
```


```python
data.head(10)
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
    <tr>
      <th>5</th>
      <td>6</td>
      <td>claim</td>
      <td>8972200955</td>
      <td>35</td>
      <td>someone shared with me that gross domestic pro...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>336647.0</td>
      <td>175546.0</td>
      <td>62303.0</td>
      <td>4293.0</td>
      <td>1857.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>claim</td>
      <td>4958886992</td>
      <td>16</td>
      <td>someone shared with me that elvis presley has ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>750345.0</td>
      <td>486192.0</td>
      <td>193911.0</td>
      <td>8616.0</td>
      <td>5446.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>claim</td>
      <td>2270982263</td>
      <td>41</td>
      <td>someone shared with me that the best selling s...</td>
      <td>not verified</td>
      <td>active</td>
      <td>547532.0</td>
      <td>1072.0</td>
      <td>50.0</td>
      <td>22.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>claim</td>
      <td>5235769692</td>
      <td>50</td>
      <td>someone shared with me that about half of the ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>24819.0</td>
      <td>10160.0</td>
      <td>1050.0</td>
      <td>53.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>claim</td>
      <td>4660861094</td>
      <td>45</td>
      <td>someone shared with me that it would take a 50...</td>
      <td>verified</td>
      <td>active</td>
      <td>931587.0</td>
      <td>171051.0</td>
      <td>67739.0</td>
      <td>4104.0</td>
      <td>2540.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (19382, 12)




```python
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
data.head(10)
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
    <tr>
      <th>5</th>
      <td>6</td>
      <td>claim</td>
      <td>8972200955</td>
      <td>35</td>
      <td>someone shared with me that gross domestic pro...</td>
      <td>not verified</td>
      <td>under review</td>
      <td>336647.0</td>
      <td>175546.0</td>
      <td>62303.0</td>
      <td>4293.0</td>
      <td>1857.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>claim</td>
      <td>4958886992</td>
      <td>16</td>
      <td>someone shared with me that elvis presley has ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>750345.0</td>
      <td>486192.0</td>
      <td>193911.0</td>
      <td>8616.0</td>
      <td>5446.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>claim</td>
      <td>2270982263</td>
      <td>41</td>
      <td>someone shared with me that the best selling s...</td>
      <td>not verified</td>
      <td>active</td>
      <td>547532.0</td>
      <td>1072.0</td>
      <td>50.0</td>
      <td>22.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>claim</td>
      <td>5235769692</td>
      <td>50</td>
      <td>someone shared with me that about half of the ...</td>
      <td>not verified</td>
      <td>active</td>
      <td>24819.0</td>
      <td>10160.0</td>
      <td>1050.0</td>
      <td>53.0</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>claim</td>
      <td>4660861094</td>
      <td>45</td>
      <td>someone shared with me that it would take a 50...</td>
      <td>verified</td>
      <td>active</td>
      <td>931587.0</td>
      <td>171051.0</td>
      <td>67739.0</td>
      <td>4104.0</td>
      <td>2540.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for duplicates
data.duplicated().sum()
```




    np.int64(0)




```python
# Check class balance
data['claim_status'].value_counts(normalize = True)
```




    claim_status
    claim      0.503458
    opinion    0.496542
    Name: proportion, dtype: float64




```python
# Extract the length of each `video_transcription_text` and add this as a column to the dataframe
data['text_length'] = data['video_transcription_text'].str.len()
```


```python
# Calculate the average text_length for claims and opinions
data[['claim_status', 'text_length']].groupby('claim_status').mean()
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
      <th>text_length</th>
    </tr>
    <tr>
      <th>claim_status</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>claim</th>
      <td>95.376978</td>
    </tr>
    <tr>
      <th>opinion</th>
      <td>82.722562</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Visualize the distribution of `text_length` for claims and opinions
sns.histplot(data = data, stat='count', multiple = 'dodge', x = 'text_length',
            kde = False, palette = 'pastel', hue = 'claim_status', 
            element = 'bars', legend = True)
plt.xlabel('video_transcription_text length (number of characters)')
plt.ylabel('Count')
plt.title('Distribution of video_transcription_text length for claims and opinions')
plt.show()
```


    
![png](output_14_0.png)
    



```python
# Create a copy of the X data
X = data.copy()
# Drop unnecessary columns
X = X.drop(['#', 'video_id'], axis = 1)
# Encode target variable
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})

# Dummy encode remaining categorical values
X = pd.get_dummies(X,
                  columns = ['verified_status', 'author_ban_status'],
                  drop_first = True)
X.head()
```

    /var/folders/g5/c80y5wdx7p32dqjlhnrw1v7m0000gn/T/ipykernel_3954/1786180476.py:6: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
      X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})





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
      <th>video_duration_sec</th>
      <th>video_transcription_text</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_length</th>
      <th>verified_status_verified</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Isolate target variable
y = X['claim_status']
```


```python
# Isolate features
X = X.drop(['claim_status'], axis = 1)
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
      <th>video_transcription_text</th>
      <th>video_view_count</th>
      <th>video_like_count</th>
      <th>video_share_count</th>
      <th>video_download_count</th>
      <th>video_comment_count</th>
      <th>text_length</th>
      <th>verified_status_verified</th>
      <th>author_ban_status_banned</th>
      <th>author_ban_status_under review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59</td>
      <td>someone shared with me that drone deliveries a...</td>
      <td>343296.0</td>
      <td>19425.0</td>
      <td>241.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>97</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
      <td>someone shared with me that there are more mic...</td>
      <td>140877.0</td>
      <td>77355.0</td>
      <td>19034.0</td>
      <td>1161.0</td>
      <td>684.0</td>
      <td>107</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31</td>
      <td>someone shared with me that american industria...</td>
      <td>902185.0</td>
      <td>97690.0</td>
      <td>2858.0</td>
      <td>833.0</td>
      <td>329.0</td>
      <td>137</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>25</td>
      <td>someone shared with me that the metro of st. p...</td>
      <td>437506.0</td>
      <td>239954.0</td>
      <td>34812.0</td>
      <td>1234.0</td>
      <td>584.0</td>
      <td>131</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>someone shared with me that the number of busi...</td>
      <td>56167.0</td>
      <td>34987.0</td>
      <td>4110.0</td>
      <td>547.0</td>
      <td>152.0</td>
      <td>128</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split the data into training and testing sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


```python
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)
```


```python
# Get shape of each training, validation, and testing set
X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape
```




    ((11450, 11), (3817, 11), (3817, 11), (11450,), (3817,), (3817,))




```python
# Instantiate the random forest classifier
RF = RandomForestClassifier(random_state = 0)
# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
            'max_features': [0.3, 0.6],
            'max_samples': [0,7],
            'min_samples_leaf': [1,2],
            'min_samples_split': [2,3],
            'n_estimators': [75, 100, 200],
            }

# Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']
# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(RF, cv_params, scoring = scoring, cv = 5, refit = 'recall')
```


```python
### Fit the model to the data 
count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()

count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())

X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)

rf_cv.fit(X_train_final, y_train)
```

    /opt/anaconda3/lib/python3.13/site-packages/sklearn/model_selection/_validation.py:528: FitFailedWarning: 
    360 fits failed out of a total of 720.
    The score on these train-test partitions for these parameters will be set to nan.
    If these failures are not expected, you can try to debug them by setting error_score='raise'.
    
    Below are more details about the failures:
    --------------------------------------------------------------------------------
    360 fits failed with the following error:
    Traceback (most recent call last):
      File "/opt/anaconda3/lib/python3.13/site-packages/sklearn/model_selection/_validation.py", line 866, in _fit_and_score
        estimator.fit(X_train, y_train, **fit_params)
        ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/anaconda3/lib/python3.13/site-packages/sklearn/base.py", line 1382, in wrapper
        estimator._validate_params()
        ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
      File "/opt/anaconda3/lib/python3.13/site-packages/sklearn/base.py", line 436, in _validate_params
        validate_parameter_constraints(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
            self._parameter_constraints,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            self.get_params(deep=False),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            caller_name=self.__class__.__name__,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )
        ^
      File "/opt/anaconda3/lib/python3.13/site-packages/sklearn/utils/_param_validation.py", line 98, in validate_parameter_constraints
        raise InvalidParameterError(
        ...<2 lines>...
        )
    sklearn.utils._param_validation.InvalidParameterError: The 'max_samples' parameter of RandomForestClassifier must be None, a float in the range (0.0, 1.0] or an int in the range [1, inf). Got 0 instead.
    
      warnings.warn(some_fits_failed_message, FitFailedWarning)
    /opt/anaconda3/lib/python3.13/site-packages/sklearn/model_selection/_search.py:1108: UserWarning: One or more of the test scores are non-finite: [       nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95144105 0.94777293 0.94253275 0.95196507 0.94917031 0.9431441
     0.95624454 0.9530131  0.94716157 0.95624454 0.9530131  0.94716157
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95519651 0.95048035 0.94576419 0.95519651 0.95056769 0.94585153
     0.95537118 0.95248908 0.94768559 0.95537118 0.95248908 0.94768559
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95144105 0.94777293 0.94253275 0.95196507 0.94917031 0.9431441
     0.95624454 0.9530131  0.94716157 0.95624454 0.9530131  0.94716157
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95519651 0.95048035 0.94576419 0.95519651 0.95056769 0.94585153
     0.95537118 0.95248908 0.94768559 0.95537118 0.95248908 0.94768559
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95144105 0.94777293 0.94253275 0.95196507 0.94917031 0.9431441
     0.95624454 0.9530131  0.94716157 0.95624454 0.9530131  0.94716157
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95519651 0.95048035 0.94576419 0.95519651 0.95056769 0.94585153
     0.95537118 0.95248908 0.94768559 0.95537118 0.95248908 0.94768559]
      warnings.warn(
    /opt/anaconda3/lib/python3.13/site-packages/sklearn/model_selection/_search.py:1108: UserWarning: One or more of the test scores are non-finite: [nan nan nan nan nan nan nan nan nan nan nan nan  1.  1.  1.  1.  1.  1.
      1.  1.  1.  1.  1.  1. nan nan nan nan nan nan nan nan nan nan nan nan
      1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. nan nan nan nan nan nan
     nan nan nan nan nan nan  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
     nan nan nan nan nan nan nan nan nan nan nan nan  1.  1.  1.  1.  1.  1.
      1.  1.  1.  1.  1.  1. nan nan nan nan nan nan nan nan nan nan nan nan
      1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. nan nan nan nan nan nan
     nan nan nan nan nan nan  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
      warnings.warn(
    /opt/anaconda3/lib/python3.13/site-packages/sklearn/model_selection/_search.py:1108: UserWarning: One or more of the test scores are non-finite: [       nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.90403585 0.89678822 0.88643447 0.90507122 0.89954952 0.88764285
     0.91352962 0.90714436 0.89558177 0.91352962 0.90714436 0.89558177
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.91145693 0.90213781 0.89282003 0.91145693 0.90231037 0.89299274
     0.9118031  0.90610764 0.89661774 0.9118031  0.90610764 0.89661774
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.90403585 0.89678822 0.88643447 0.90507122 0.89954952 0.88764285
     0.91352962 0.90714436 0.89558177 0.91352962 0.90714436 0.89558177
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.91145693 0.90213781 0.89282003 0.91145693 0.90231037 0.89299274
     0.9118031  0.90610764 0.89661774 0.9118031  0.90610764 0.89661774
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.90403585 0.89678822 0.88643447 0.90507122 0.89954952 0.88764285
     0.91352962 0.90714436 0.89558177 0.91352962 0.90714436 0.89558177
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.91145693 0.90213781 0.89282003 0.91145693 0.90231037 0.89299274
     0.9118031  0.90610764 0.89661774 0.9118031  0.90610764 0.89661774]
      warnings.warn(
    /opt/anaconda3/lib/python3.13/site-packages/sklearn/model_selection/_search.py:1108: UserWarning: One or more of the test scores are non-finite: [       nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.94957772 0.94557905 0.9397965  0.95014595 0.94711086 0.94047425
     0.95480301 0.95130832 0.94491067 0.95480301 0.95130832 0.94491067
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95364721 0.94852655 0.94337379 0.95364721 0.9486222  0.94347024
     0.95385306 0.95072522 0.94548409 0.95385306 0.95072522 0.94548409
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.94957772 0.94557905 0.9397965  0.95014595 0.94711086 0.94047425
     0.95480301 0.95130832 0.94491067 0.95480301 0.95130832 0.94491067
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95364721 0.94852655 0.94337379 0.95364721 0.9486222  0.94347024
     0.95385306 0.95072522 0.94548409 0.95385306 0.95072522 0.94548409
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.94957772 0.94557905 0.9397965  0.95014595 0.94711086 0.94047425
     0.95480301 0.95130832 0.94491067 0.95480301 0.95130832 0.94491067
            nan        nan        nan        nan        nan        nan
            nan        nan        nan        nan        nan        nan
     0.95364721 0.94852655 0.94337379 0.95364721 0.9486222  0.94347024
     0.95385306 0.95072522 0.94548409 0.95385306 0.95072522 0.94548409]
      warnings.warn(





<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-4 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=0),
             param_grid={&#x27;max_depth&#x27;: [5, 7, None], &#x27;max_features&#x27;: [0.3, 0.6],
                         &#x27;max_samples&#x27;: [0, 7], &#x27;min_samples_leaf&#x27;: [1, 2],
                         &#x27;min_samples_split&#x27;: [2, 3],
                         &#x27;n_estimators&#x27;: [75, 100, 200]},
             refit=&#x27;recall&#x27;, scoring=[&#x27;accuracy&#x27;, &#x27;precision&#x27;, &#x27;recall&#x27;, &#x27;f1&#x27;])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=0),
             param_grid={&#x27;max_depth&#x27;: [5, 7, None], &#x27;max_features&#x27;: [0.3, 0.6],
                         &#x27;max_samples&#x27;: [0, 7], &#x27;min_samples_leaf&#x27;: [1, 2],
                         &#x27;min_samples_split&#x27;: [2, 3],
                         &#x27;n_estimators&#x27;: [75, 100, 200]},
             refit=&#x27;recall&#x27;, scoring=[&#x27;accuracy&#x27;, &#x27;precision&#x27;, &#x27;recall&#x27;, &#x27;f1&#x27;])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: RandomForestClassifier</div></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(max_depth=5, max_features=0.3, max_samples=7,
                       min_samples_leaf=2, n_estimators=75, random_state=0)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RandomForestClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestClassifier.html">?<span>Documentation for RandomForestClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>RandomForestClassifier(max_depth=5, max_features=0.3, max_samples=7,
                       min_samples_leaf=2, n_estimators=75, random_state=0)</pre></div> </div></div></div></div></div></div></div></div></div>




```python
# Examine best recall score
rf_cv.best_score_
```




    np.float64(0.9135296195129803)




```python
# Examine best parameters
rf_cv.best_params_
```




    {'max_depth': 5,
     'max_features': 0.3,
     'max_samples': 7,
     'min_samples_leaf': 2,
     'min_samples_split': 2,
     'n_estimators': 75}




```python
# Access the GridSearch results and convert it to a pandas df
rf_results_df = pd.DataFrame(rf_cv.cv_results_)
# Examine the GridSearch results df at column `mean_test_precision` in the best index
rf_results_df['mean_test_precision'][rf_cv.best_index_]
```




    np.float64(1.0)




```python
# Instantiate the XGBoost classifier
xgb = XGBClassifier(objetive = 'binary: logistic', random_state = 0)
# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
            'min_child_weight': [3,5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [300, 500]}

# Define a list of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']
# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring = scoring, cv = 5, refit = 'recall')

```


```python
# Fit the model to the data
xgb_cv.fit(X_train_final, y_train)
```

    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:15] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:16] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:16] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:16] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:16] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:16] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:17] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:17] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:17] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:17] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:18] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:18] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:18] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:18] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:18] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:18] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:19] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:19] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:19] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:19] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:20] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:20] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:20] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:20] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:21] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:21] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:21] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:21] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:22] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:22] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:23] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:23] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:23] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:23] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:24] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:24] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:24] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:24] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:25] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:25] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:25] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:25] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:26] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:26] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:26] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:26] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:27] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:27] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:27] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:28] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:28] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:28] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:28] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:29] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:29] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:29] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:29] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:30] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:30] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:30] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:31] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:31] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:31] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:31] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:31] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:32] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:32] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:32] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:32] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:33] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:33] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:33] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:33] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:33] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:34] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:34] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:34] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:34] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:35] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:35] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:35] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:35] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:36] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:36] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:36] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:36] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:37] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:37] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:38] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:38] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:38] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:39] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:39] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:39] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:39] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:40] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:40] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:40] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:40] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:41] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:41] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:41] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:41] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:42] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:42] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:42] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:43] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:43] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:44] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:44] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:45] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:45] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:45] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:45] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:46] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:46] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:46] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:47] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:48] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:48] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/anaconda3/lib/python3.13/site-packages/xgboost/training.py:200: UserWarning: [17:01:49] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:793: 
    Parameters: { "objetive" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)





<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-2 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None, device=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=True, eval_metric=None,
                                     feature_types=None, feature_weights=None,
                                     gamma=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None...
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     multi_strategy=None, n_estimators=None,
                                     n_jobs=None, num_parallel_tree=None, ...),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1], &#x27;max_depth&#x27;: [4, 8, 12],
                         &#x27;min_child_weight&#x27;: [3, 5],
                         &#x27;n_estimators&#x27;: [300, 500]},
             refit=&#x27;recall&#x27;, scoring=[&#x27;accuracy&#x27;, &#x27;precision&#x27;, &#x27;recall&#x27;, &#x27;f1&#x27;])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GridSearchCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html">?<span>Documentation for GridSearchCV</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=5,
             estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None, device=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=True, eval_metric=None,
                                     feature_types=None, feature_weights=None,
                                     gamma=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None...
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     multi_strategy=None, n_estimators=None,
                                     n_jobs=None, num_parallel_tree=None, ...),
             param_grid={&#x27;learning_rate&#x27;: [0.01, 0.1], &#x27;max_depth&#x27;: [4, 8, 12],
                         &#x27;min_child_weight&#x27;: [3, 5],
                         &#x27;n_estimators&#x27;: [300, 500]},
             refit=&#x27;recall&#x27;, scoring=[&#x27;accuracy&#x27;, &#x27;precision&#x27;, &#x27;recall&#x27;, &#x27;f1&#x27;])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>best_estimator_: XGBClassifier</div></div></label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=True, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=4,
              max_leaves=None, min_child_weight=5, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=300,
              n_jobs=None, num_parallel_tree=None, ...)</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>XGBClassifier</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://xgboost.readthedocs.io/en/release_3.3.0/python/python_api.html#xgboost.XGBClassifier">?<span>Documentation for XGBClassifier</span></a></div></label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=True, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=4,
              max_leaves=None, min_child_weight=5, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=300,
              n_jobs=None, num_parallel_tree=None, ...)</pre></div> </div></div></div></div></div></div></div></div></div>




```python
# Examine best recall score
xgb_cv.best_score_
```




    np.float64(0.9898176171763818)




```python
# Examine best parameters
xgb_cv.best_params_
```




    {'learning_rate': 0.1,
     'max_depth': 4,
     'min_child_weight': 5,
     'n_estimators': 300}




```python
# Access the GridSearch results and convert it to a pandas df
xgb_results_df = pd.DataFrame(xgb_cv.cv_results_)
# Examine the GridSearch results df at column `mean_test_precision` in the best index
xgb_results_df['mean_test_precision'][xgb_cv.best_index_]
```




    np.float64(0.9989540875321314)




```python
# Use the random forest "best estimator" model to get predictions on the validation set
validation_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()

validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())
X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis = 1)
y_pred = rf_cv.best_estimator_.predict(X_val_final)
```

Display the predictions on the validation set.


```python
# Display the predictions on the validation set
y_pred
```




    array([1, 0, 1, ..., 1, 1, 1])




```python
# Display the true labels of the validation set
y_val
```




    5846     1
    12058    0
    2975     1
    8432     1
    6863     1
            ..
    6036     1
    6544     1
    2781     1
    6426     1
    4450     1
    Name: claim_status, Length: 3817, dtype: int64



Create a confusion matrix to visualize the results of the classification model.


```python
# Create a confusion matrix to visualize the results of the classification model

# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)
# Create display of confusion matrix using ConfusionMatrixDisplay()
log_disp = ConfusionMatrixDisplay(confusion_matrix = log_cm, display_labels = None)
# Plot confusion matrix
log_disp.plot()
# Display plot
plt.show()
```


    
![png](output_36_0.png)
    


Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the model.
<br> </br>

**Note:** In other labs there was a custom-written function to extract the accuracy, precision, recall, and F<sub>1</sub> scores from the GridSearchCV report and display them in a table. You can also use scikit-learn's built-in [`classification_report()`](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) function to obtain a similar table of results.


```python
# Create classification report for random forest model
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names = target_labels))
```

                  precision    recall  f1-score   support
    
         opinion       0.91      1.00      0.95      1892
           claim       1.00      0.91      0.95      1925
    
        accuracy                           0.95      3817
       macro avg       0.96      0.95      0.95      3817
    weighted avg       0.96      0.95      0.95      3817
    



```python
# Use the best estimator to predict on the validation data
y_pred = xgb_cv.best_estimator_.predict(X_val_final)
```


```python
# Compute values for confusion matrix
log_cm = confusion_matrix(y_val, y_pred)
# Create display of confusion matrix using ConfusionMatrixDisplay()
log_disp = ConfusionMatrixDisplay(confusion_matrix = log_cm, display_labels = None)

# Plot confusion matrix
log_disp.plot()
# Display plot
plt.title('XGBoost - validation set')
plt.show()
```


    
![png](output_40_0.png)
    



```python
# Create a classification report
target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))
```

                  precision    recall  f1-score   support
    
         opinion       0.99      1.00      0.99      1892
           claim       1.00      0.99      0.99      1925
    
        accuracy                           0.99      3817
       macro avg       0.99      0.99      0.99      3817
    weighted avg       0.99      0.99      0.99      3817
    



```python
test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()

test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text'] ).reset_index(drop=True), test_count_df], axis=1)

y_pred = rf_cv.best_estimator_.predict(X_test_final)
```


```python
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred)
# Create display of confusion matrix using ConfusionMatrixDisplay()

log_disp = ConfusionMatrixDisplay(confusion_matrix = log_cm, display_labels = None)
# Plot confusion matrix
log_disp.plot()
# Display plot
plt.title('Random forest - test set')
plt.show()
```


    
![png](output_43_0.png)
    



```python
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()
```


    
![png](output_44_0.png)
    

