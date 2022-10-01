# Case study on auction for showing advertisement to random non-identifiable users

**Tasks:**
- Problem 1: For a given aggregated data set the expected win rate shall be estimated
- Problem 2: If the maximum budget of the advertisers would be 0.5, what would be the best bid valuation
    
**Steps:**
- Check the data and calculate the rates as amount of events with win divided by amount of wins
- Calculate max Win per bid_price and take into account the expected win rate
- Chose the best bid_price to maximize the win * expected win rate
- If there is a clear trend or tendency in the data, fit the win * expected win rate as function of bid_price, to be able to give best bid_prices with are not included in the underlying data

**Downfalls:**
- The data lacks a resonable discretization, it is not clear where the decay of the win * expected win rate occurs.
- Therefore, it is not really useful to fit to any function, here the weigthed linear fit shall only show the general idea

**Results:**
- Problem 1: see table below
- Problem 2: by the given data, the choice is a bid_price of 0.10, as it has the highest gain considering its expected win rate.
- Real application: Set up a dynamic approach, which tracts the expected and realized wins and modifies the strategy as competitors as well as budgets of competitors change and also there strategies. For fast reception and ability to counter changes, A/B-Test should be used to also bid on not optimal prices to estimate possible needed increases or possible decreases for more win extraction.


```python
# Libraries
import numpy as np
import pandas as pd
#import scipy as sp
# for clopper pearson intervals
from statsmodels.stats.proportion import proportion_confint
# for 2. problem and weigthed linear regression
from sklearn.linear_model import LinearRegression

```


```python
# Load data
data = pd.read_json('./data.json')
```


```python
# group by for sum of all events and only wins per bid_price
aggDat = data.groupby('bid_price').apply(lambda s: pd.Series({
            'eventsAgg': s['events'].sum(),
            'eventsWinAgg': (s['events'] * s['win']).sum(),
        }
    )   
)

# group by for 1. problem expected win rate: expWinRate
aggDat['expWinRate'] = aggDat['eventsWinAgg'] / aggDat['eventsAgg']

# add some clopper pearson confidence bands
aggDat['confInt'] = aggDat.apply(
    lambda x: proportion_confint(
        int(
            x['eventsWinAgg']
        ),
        int(
            x['eventsAgg']
        ),
        alpha=0.05,
        method='normal'
    ),
    axis= 1,
)

# as confInt being a tuple, split it and round for better presentation
aggDat['lowerConfInt'] = (pd.DataFrame(aggDat['confInt'].tolist()).apply(lambda x : round(x,3))[0]).tolist()
aggDat['upperConfInt'] = (pd.DataFrame(aggDat['confInt'].tolist()).apply(lambda x : round(x,3))[1]).tolist()

del aggDat['confInt']

# add weights for later weigthed linear fit
aggDat['weight'] = aggDat['eventsAgg'] / aggDat['eventsAgg'].sum()

# get rid of index from grouping, as it will be needed as column
aggDat = aggDat.reset_index()

# add target for minimization
aggDat['win'] = 0.5 - aggDat['bid_price'] # possible win, if 0.5 is max budget 
aggDat['winProb'] = aggDat['win'] * aggDat['expWinRate'] # taking into account probabiltiy to win times win

aggDat.to_json('finalTable.json')
```


```python
# 1. Problem: see bid_price vs expWinRate
aggDat
# 2. Problem: for discrete given bid_prices best valutation is 0.10 with the highest winProb and win of 0.40
# But, in real bidding the competors, there algorithms and budgets change, therefore a more dynamical approch must be utilized in real life
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
      <th>bid_price</th>
      <th>eventsAgg</th>
      <th>eventsWinAgg</th>
      <th>expWinRate</th>
      <th>lowerConfInt</th>
      <th>upperConfInt</th>
      <th>weight</th>
      <th>win</th>
      <th>winProb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01</td>
      <td>100000</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>8.911774e-03</td>
      <td>0.49</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.10</td>
      <td>10000</td>
      <td>3000</td>
      <td>0.3</td>
      <td>0.291</td>
      <td>0.309</td>
      <td>8.911774e-04</td>
      <td>0.40</td>
      <td>0.120</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.20</td>
      <td>10000000</td>
      <td>2000000</td>
      <td>0.2</td>
      <td>0.200</td>
      <td>0.200</td>
      <td>8.911774e-01</td>
      <td>0.30</td>
      <td>0.060</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.40</td>
      <td>1000000</td>
      <td>300000</td>
      <td>0.3</td>
      <td>0.299</td>
      <td>0.301</td>
      <td>8.911774e-02</td>
      <td>0.10</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.50</td>
      <td>100000</td>
      <td>20000</td>
      <td>0.2</td>
      <td>0.198</td>
      <td>0.202</td>
      <td>8.911774e-03</td>
      <td>0.00</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.75</td>
      <td>10000</td>
      <td>3000</td>
      <td>0.3</td>
      <td>0.291</td>
      <td>0.309</td>
      <td>8.911774e-04</td>
      <td>-0.25</td>
      <td>-0.075</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.00</td>
      <td>1000</td>
      <td>600</td>
      <td>0.6</td>
      <td>0.570</td>
      <td>0.630</td>
      <td>8.911774e-05</td>
      <td>-0.50</td>
      <td>-0.300</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.00</td>
      <td>100</td>
      <td>70</td>
      <td>0.7</td>
      <td>0.610</td>
      <td>0.790</td>
      <td>8.911774e-06</td>
      <td>-1.50</td>
      <td>-1.050</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.00</td>
      <td>10</td>
      <td>8</td>
      <td>0.8</td>
      <td>0.552</td>
      <td>1.000</td>
      <td>8.911774e-07</td>
      <td>-4.50</td>
      <td>-3.600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.00</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.000</td>
      <td>1.000</td>
      <td>8.911774e-08</td>
      <td>-8.50</td>
      <td>-8.500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot, (lowerConfInt for bid_price 9 does not make sense - please ignore - not further corrected)
aggDat.plot(x = 'bid_price', y=['expWinRate','lowerConfInt','upperConfInt'], kind='line', logx = True)  
aggDat.plot(x = 'bid_price', y=['win','winProb'], kind='line',logx=True) 

```




    <AxesSubplot:xlabel='bid_price'>




    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



```python
# 2. problem: task: fit winProb on bid_price - for possible interpolation for not given bid_prices
# the regression is of course not really useful, as we see in aggDat, that the fit will be dominated such, that a lower bid_price always ensures a higher win - the data is insufficent discretised and it is not clear, where the expWinRate starts to reduce below 0.1 as bid_price to assume a different polynominal fit 
reg = LinearRegression().fit(aggDat['bid_price'].to_numpy().reshape(-1, 1), aggDat['winProb'].to_numpy().reshape(-1, 1), aggDat['weight'])

print('Coef = ', reg.coef_)
print('Intercept = ', reg.intercept_)

bidPrices = np.linspace(0.01, 0.5, num=100)

predicts = pd.DataFrame({
    'bidPrices':  bidPrices,
    'winProbPredicted': reg.predict(bidPrices.reshape(-1, 1)).reshape(-1)
})


predicts.plot(x = 'bidPrices', y=['winProbPredicted'], kind='line') 
aggDat.iloc[0:5,].plot(x = 'bid_price', y=['winProb'], kind='line') 


```

    Coef =  [[-0.13944029]]
    Intercept =  [0.08672309]
    




    <AxesSubplot:xlabel='bid_price'>




    
![png](output_7_2.png)
    



    
![png](output_7_3.png)
    



```python

```
