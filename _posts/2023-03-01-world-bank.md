---
layout: splash
permalink: /world-bank/
title: "World Bank Data"
header:
  overlay_image: /assets/images/world-bank/world-bank-splash.jpeg
excerpt: "Analyzing some of the data from the World Bank"
---

```python
def my_plot(fig):
    # to incorporate the result in the github.io page
    from IPython.display import HTML
    fig.show()
    HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
```


```python
import plotly.express as px

df = px.data.stocks(indexed=True)-1
fig = px.area(df, facet_col="company", facet_col_wrap=2)
my_plot(fig)
```


