# **TikTok Project**
**Course 2 - Go Beyond the Numbers: Translate Data into Insights**


```python
# Import packages for data manipulation
import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")
```


```python
# Display and examine the first few rows of the dataframe
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
# Get the size of the data
data.size
```




    232584




```python
# Get the shape of the data
data.shape
```




    (19382, 12)




```python
# Get basic information about the data
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


Generate a table of descriptive statistics, using `.describe()`.


```python
# Generate a table of descriptive statistics
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
# Create a boxplot to visualize distribution of `video_duration_sec`
sns.boxplot(x=data['video_duration_sec']);
plt.title('Video Duration Sec')
```




    Text(0.5, 1.0, 'Video Duration Sec')




    
![png](output_9_1.png)
    



```python
# Create a histogram
sns.histplot(data['video_duration_sec'], bins = range(0,61,5));
plt.title('Video Duration (sec)')
```




    Text(0.5, 1.0, 'Video Duration (sec)')




    
![png](output_10_1.png)
    


The distribution is uniform and videos range from 5 to 60 seconds.


```python
sns.boxplot(x = data['video_view_count']);
plt.title('Video View Count')
```




    Text(0.5, 1.0, 'Video View Count')




    
![png](output_12_1.png)
    


Create a histogram of the values in the `video_view_count` column to further explore the distribution of this variable.


```python
sns.histplot(data['video_view_count'], bins = range(0,10**6+1, 10**5));
plt.title('Video View Count')
```




    Text(0.5, 1.0, 'Video View Count')




    
![png](output_14_1.png)
    


The distribution is right-skewed.


```python
# Create a boxplot to visualize distribution of `video_like_count`
sns.boxplot(x = data['video_like_count']);
plt.title('Video like Count')
```




    Text(0.5, 1.0, 'Video like Count')




    
![png](output_16_1.png)
    



```python
sns.histplot(data['video_like_count'], bins = range(0,600000, 100000));
```


    
![png](output_17_0.png)
    


right-skewedThis distribution is right skewed.


```python
# Create a boxplot to visualize distribution of `video_comment_count`
plt.title('Video Comment Count')
sns.boxplot(x = data['video_comment_count']);

```


    
![png](output_19_0.png)
    


Create a histogram of the values in the `video_comment_count` column to further explore the distribution of this variable.


```python
sns.histplot(data['video_comment_count'], bins = range(0,4000,200));
```


    
![png](output_21_0.png)
    



```python
# Create a boxplot to visualize distribution of `video_share_count`
sns.boxplot(x = data['video_share_count']);
plt.title('video share count')
```




    Text(0.5, 1.0, 'video share count')




    
![png](output_22_1.png)
    



```python
sns.histplot(data['video_share_count'], bins = range(0, 175000, 10000))
```




    <Axes: xlabel='video_share_count', ylabel='Count'>




    
![png](output_23_1.png)
    


right-skewedMost videos had under 25000 views. The distribution is right skewed.


```python
# Create a boxplot to visualize distribution of `video_download_count`
sns.boxplot(x = data['video_download_count']);
plt.title('video download count')
```




    Text(0.5, 1.0, 'video download count')




    
![png](output_25_1.png)
    



```python
sns.histplot(data['video_download_count'], bins = range(0, 14000, 500));
plt.title('Video Download Count')
```




    Text(0.5, 1.0, 'Video Download Count')




    
![png](output_26_1.png)
    


fewerright-skewedA majority of videos receive less than 20000 downloads. The distribution is heavily right skewed.


```python
sns.histplot(data = data,
             x = 'claim_status',
            hue = 'verified_status',
            multiple = 'dodge')
plt.title('Claims by verification status Histogram ')
```




    Text(0.5, 1.0, 'Claims by verification status Histogram ')




    
![png](output_28_1.png)
    


There are far more non-verified users than verified users. In addition, verified users have more opinion videos than claim videos. On the other hand, non-verified users have more claim videos than opinion videos. 


```python
# Create a histogram
sns.histplot(data = data,
            x = 'claim_status',
            hue = 'author_ban_status',
            multiple = 'dodge',
            palette = {'active': 'green', 'under review': 'orange', 'banned': 'red'})
plt.title('Claim status by Author ban status')
```




    Text(0.5, 1.0, 'Claim status by Author ban status')




    
![png](output_30_1.png)
    


There are far more banned authors with claims than those with opinions. Additionally, there are far more active opinion users.


```python
# Create a bar plot
ban_status_count = data.groupby(['author_ban_status']).median(numeric_only = True).reset_index()
sns.barplot(data = ban_status_count,
           x = 'author_ban_status',
           y = 'video_view_count',
           palette = {'active': 'green', 'under review': 'orange', 'banned': 'red'})
plt.title('Author Ban Status Median View')
```

    /var/folders/g5/c80y5wdx7p32dqjlhnrw1v7m0000gn/T/ipykernel_3508/1380570784.py:3: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(data = ban_status_count,





    Text(0.5, 1.0, 'Author Ban Status Median View')




    
![png](output_32_2.png)
    


The median view count for non-active authors are significantly larger than that of active users. As authors who post more claims get banned and aggregate more views, we can assume that video view count is a good indicator of claim status.


```python
# Calculate the median view count for claim status.
data.groupby('claim_status')['video_view_count'].median()
```




    claim_status
    claim      501555.0
    opinion      4953.0
    Name: video_view_count, dtype: float64




```python
# Create a pie graph
plt.pie(data.groupby('claim_status')['video_view_count'].sum(), labels = ['claim', 'opinion'])
plt.title('Total views by video claim status')
```




    Text(0.5, 1.0, 'Total views by video claim status')




    
![png](output_35_1.png)
    


**Question:** What do you notice about the overall view count for claim status?

Video views are mostly made of claim videos.


```python
### YOUR CODE HERE ###

columns = ['video_view_count', 'video_like_count',
          'video_share_count', 'video_download_count',
          'video_comment_count']
for n in columns:
    q1 = data[n].quantile(0.25)
    q3 = data[n].quantile(0.75)
    iqr = q3 - q1
    median = data[n].median()
    outlier = median + 1.5*iqr
    
    outlier_count = (data[n] > outlier).sum()
    print(f'Number of outliers in {n} is  {outlier_count}')
```

    Number of outliers in video_view_count is  2343
    Number of outliers in video_like_count is  3468
    Number of outliers in video_share_count is  3732
    Number of outliers in video_download_count is  3733
    Number of outliers in video_comment_count is  3882


#### **Scatterplot**


```python
# Create a scatterplot of `video_view_count` versus `video_like_count` according to 'claim_status'
sns.scatterplot(x = data['video_view_count'],
               y = data['video_like_count'],
               hue = data['claim_status'])
plt.title('Video View Count vs Video Like Count')
plt.show()
```


    
![png](output_39_0.png)
    



```python
# Create a scatterplot of ``video_view_count` versus `video_like_count` for opinions only
opinion = data[data['claim_status'] == 'opinion']
sns.scatterplot(x = opinion['video_view_count'],
               y = opinion['video_like_count'],
               hue = opinion['claim_status'])
plt.title('Video View Count vs Video Like Count for opinion videos')
plt.show()

```


    
![png](output_40_0.png)
    


Visualizations helped me process correlations in the data more easily. Graphs help illustrate correlations far more than a correlation value. To a stakeholder with little understanding of data, visualization can promote understanding and allow for better business decisions.

