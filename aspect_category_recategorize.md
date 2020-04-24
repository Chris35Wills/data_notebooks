# Aspect values to categories

When dealing with terrain aspect values, you might find yourself needing to categorize them into sectors e.g. 10 deg N = North, 24 deg North = North-east, 90 deg North = East etc. This is also something to consider when using such data in a machine learning workflow too as you may find you need to [encode the categories](https://www.dataschool.io/encoding-categorical-features-in-python/) rather than use the raw aspect orientations (rescaling aspect orientations between 0 and 1 doesn;t make sense....)

Below is a helper function that does the aspect category mapping for you. You just need to pass it an input array such as a pandas series.


```python
import pandas as pd
import numpy as np

def categorize_aspect(aspect_array, check_output=True):
    """Recategorizes numerical aspect values to quadrants (N/NE/E etc.)
    
    Takes in an e.g. pandas data frame series (expected to be 1D)
    e.g. categorize_aspect(dataframe['your_column'])
    
    North:     0° – 22.5°
    Northeast: 22.5° – 67.5°
    East:      67.5° – 112.5°
    Southeast: 112.5° – 157.5°
    South:     157.5° – 202.5°
    Southwest: 202.5° – 247.5°
    West:      247.5° – 292.5°
    Northwest: 292.5° – 337.5°
    North:     337.5° – 360°

    Returns: numpy ndarray
    """
    
    bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5,
            247.5, 292.5, 337.5, 360]
    names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N2'] 
            # we'll have 2 N quadrants so use N2 for the second as they have to be unique 

    aspect_cat = pd.cut(aspect_array, bins, labels=names)
    aspect_cat[aspect_cat=='N2']='N' # merge N2 cats with N
    
    # Check values have mapped correctly - prints to standard output
    if check_output:
        random_indxs=np.random.randint(low=1, high=aspect_cat.shape[0], size=20) #random values within shape of series
        before=aspect_array[random_indxs]
        after=aspect_cat[random_indxs]
        print("Aspect (deg N) | Aspect quadrant")
        for i in range(0,len(before)):
            print("%0.2f  | %s" %(before.iloc[i],after.iloc[i])) 

    return(aspect_cat)

# Apply it (here with a pandas series)
aspect_series=pd.Series([52.93,38.35,194.84,299.53,24.81,259.13,66.14,292.85,195.72,27.85])
aspect_categories=categorize_aspect(aspect_series)
```

    Aspect (deg N) | Aspect quadrant
    195.72  | S
    259.13  | W
    194.84  | S
    194.84  | S
    292.85  | NW
    38.35  | NE
    24.81  | NE
    38.35  | NE
    195.72  | S
    195.72  | S
    292.85  | NW
    66.14  | NE
    66.14  | NE
    38.35  | NE
    194.84  | S
    195.72  | S
    292.85  | NW
    292.85  | NW
    194.84  | S
    66.14  | NE
    


```python

```
