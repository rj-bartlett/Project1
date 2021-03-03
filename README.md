### Part 1 - Obtaining Data and Creating the Model
Here is a [link](https://rj-bartlett.github.io/Project1Cleaning/) to the code that was used to get the data from Zillow. 

After the data was collected and organized, here is the code that I used to create the model and subsequent operations. A link to the write up and plot can be found [here]().
```python
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

homes = pd.read_csv('out.csv')

homes[['price_scale']] = homes['price']/100000 
homes[['sqft_scale']] = homes['sqft']/1000 

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[3])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x1 = np.array(homes.iloc[:,2], dtype =float)
x2 = np.array(homes.iloc[:,3], dtype =float)
x3 = np.array(homes.iloc[:,6], dtype =float)
xs = np.stack([x1, x2, x3], axis=1)
ys = np.array(homes.iloc[:,5], dtype =float)

history = model.fit(xs, ys, epochs=500)

p = model.predict(xs)
homes[['predict']] = p*100000
homes[['diff']] = homes['predict']*100000 - homes['price']
```

#### Code for actual/predicted price plot
``` python
plt.scatter(homes['price'], homes['predict'])
plt.xlabel('asking price')
plt.ylabel('predictions')
plt.legend()
plt.show()
```

#### Code for heatmaps
1. Actual
``` python
homes_df = homes[['price_scale','no_beds','baths','sqft_scale']]
# Rename columns
homes_df.columns = ['Scaled Price', '# Bedrooms', '# Bathrooms', 'Sqft Scaled']
homes_corr = homes_df.corr()

# Plot
import seaborn as sns
plt.figure(figsize = (10,10))
sns.heatmap(homes_corr,
            cmap = 'Blues',
            vmin = 0, 
            vmax = 1, 
            annot = True, 
            fmt = '.2f',
            annot_kws = {'size':12},
            linewidths = .05)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
```

2. Predicted
``` python
homes_df = homes[['predict','no_beds','baths','sqft_scale']]
# Rename columns
homes_df.columns = ['Predicted Price', '# Bedrooms', '# Bathrooms', 'Sqft Scaled']
homes_corr = homes_df.corr()

# Plot
import seaborn as sns
plt.figure(figsize = (10,10))
sns.heatmap(homes_corr,
            cmap = 'Reds',
            vmin = 0, 
            vmax = 1, 
            annot = True, 
            fmt = '.2f',
            annot_kws = {'size':12},
            linewidths = .05)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
```

#### Code for PCP
``` python
# Rounding house prices to reduce Legend
homes_df['Rounded Price'] = homes_df['Scaled Price'].round(0)
homes_df2 = homes_df.sort_values(by = ['Rounded Price'])
# Grouping prices > 1mil 
idx1 = homes_df2['Rounded Price'] >= 10
idx2 = homes_df2['Rounded Price'] >= 20
idx3 = homes_df2['Rounded Price'] >= 30
homes_df2.loc[idx1, 'Rounded Price'] = '10-20'
homes_df2.loc[idx2, 'Rounded Price'] = '20-30'
homes_df2.loc[idx3, 'Rounded Price'] = '30+'
#Importing Library
from pandas.plotting import parallel_coordinates as pcp
# Plot
plt.figure(figsize = (10,10))
pcp(homes_df2.drop(columns = 'Scaled Price'), 
    'Rounded Price',
    colormap = 'cool')
plt.show()
```
