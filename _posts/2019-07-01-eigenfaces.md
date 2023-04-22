---
layout: splash
permalink: /eigenfaces/
title: "Eigenfaces"
header:
  overlay_image: /assets/images/eigenfaces/eigenfaces-splash.jpeg
excerpt: "Principal Component Analysis for Image Recognition."
---

https://sandipanweb.wordpress.com/2018/01/06/eigenfaces-and-a-simple-face-detector-with-pca-svd-in-python/


```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```


```python
faces_data = fetch_olivetti_faces()
```


```python
n_samples, height, width = faces_data.images.shape
X = faces_data.data
n_features = X.shape[1]
Y = faces_data.target
n_classes = max(Y)+1

print("""
Number of samples: {}, 
Height of each image: {},
Width of each image: {},
Number of input features: {},
Number of output classes: {}""".format(n_samples,height,
                                        width,n_features,n_classes))
```

    
    Number of samples: 400, 
    Height of each image: 64,
    Width of each image: 64,
    Number of input features: 4096,
    Number of output classes: 40
    


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)
```


```python
mean_image = np.mean(X_train, axis=0)
plt.imshow(mean_image.reshape((64, 64)), cmap=plt.cm.gray)
```




    <matplotlib.image.AxesImage at 0x26d29aab190>




    
![png](/assets/images/eigenfaces/eigenfaces-1.png)
    



```python
fig, axes = plt.subplots(figsize=(10, 50), ncols=10, nrows=40)
for i in range(40):
    current = X_train[y_train == i]
    for j in range(min(10, len(current))):
        axes[i, j].imshow(current[j].reshape((64, 64)), cmap=plt.cm.gray)
        axes[i, j].set_title(f'Person #{i}', fontsize=8)
    for j in range(10):
        axes[i, j].axis('off')
fig.tight_layout()
```


    
![png](/assets/images/eigenfaces/eigenfaces-2.png)
    



```python
n_components = 300
```


```python
pca = PCA(n_components=n_components, whiten=False).fit(X_train)
```


```python
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.axhline(y=0.5, linestyle='dashed', color='orange')
plt.axhline(y=0.75, linestyle='dashed', color='salmon')
plt.axhline(y=0.95, linestyle='dashed', color='red')
plt.grid();
```


    
![png](/assets/images/eigenfaces/eigenfaces-3.png)
    



```python
eigenfaces = pca.components_.reshape((n_components, height, width))
```


```python
fig, axes = plt.subplots(figsize=(20, 20), nrows=10, ncols=10)
axes = axes.flatten()
for i in range(100):
    axes[i].imshow(eigenfaces[i].reshape(64, 64), cmap=plt.cm.gray)
    axes[i].axis('off')
    axes[i].set_title(f'Eigenface #{i}', fontsize=10)
fig.tight_layout()
```


    
![png](/assets/images/eigenfaces/eigenfaces-4.png)
    



```python
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```


```python
def plot_reduced(i):
    fig, axes = plt.subplots(figsize=(20, 8), ncols=5)
    axes[0].imshow(X_test[i].reshape(64, 64), cmap=plt.cm.gray)
    axes[0].set_title(f'Original #{i}')
    axes[0].axis('off')
    for j, dim in enumerate([25, 50, 100, 200]):
        reduced = np.dot(X_test_pca[i, :dim], pca.components_[:dim, :]) + pca.mean_
        axes[j + 1].imshow(reduced.reshape(64, 64), cmap=plt.cm.gray)
        axes[j + 1].set_title(f'dim={dim}')
        axes[j + 1].axis('off')
    fig.tight_layout()
```


```python
plot_reduced(0)
```


    
![png](/assets/images/eigenfaces/eigenfaces-5.png)
    



```python
plot_reduced(4)
```


    
![png](/assets/images/eigenfaces/eigenfaces-6.png)
    



```python
plot_reduced(13)
```


    
![png](/assets/images/eigenfaces/eigenfaces-7.png)
    



```python
plot_reduced(32)
```


    
![png](/assets/images/eigenfaces/eigenfaces-8.png)
    



```python
plot_reduced(40)
```


    
![png](/assets/images/eigenfaces/eigenfaces-9.png)
    



```python
plot_reduced(70)
```


    
![png](/assets/images/eigenfaces/eigenfaces-10.png)
    



```python
print("Current shape of input data matrix: ", X_train_pca.shape)
```

    Current shape of input data matrix:  (300, 150)
    


```python
svm_classifier = SVMClassifier(n_neighbors = 5)
svm_classifier.fit(X_train_pca, y_train)

y_pred_test = svm_classifier.predict(X_test_pca)
correct_count = 0.0
for i in range(len(y_test)):
    if y_pred_test[i] == y_test[i]:
        correct_count += 1.0
accuracy = correct_count/float(len(y_test))
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test, labels=range(n_classes)))
```

    Accuracy: 0.58
                  precision    recall  f1-score   support
    
               0       0.00      0.00      0.00         4
               1       0.00      0.00      0.00         2
               2       1.00      0.50      0.67         2
               3       1.00      0.25      0.40         4
               4       0.43      1.00      0.60         3
               5       0.60      1.00      0.75         3
               6       0.00      0.00      0.00         1
               7       1.00      0.29      0.44         7
               8       0.50      1.00      0.67         2
               9       0.75      1.00      0.86         3
              10       1.00      0.67      0.80         3
              11       1.00      0.50      0.67         4
              12       1.00      1.00      1.00         2
              13       1.00      1.00      1.00         1
              14       0.27      1.00      0.43         3
              15       1.00      0.50      0.67         2
              17       0.67      0.67      0.67         3
              18       0.50      1.00      0.67         2
              19       0.00      0.00      0.00         1
              20       0.50      0.50      0.50         2
              21       1.00      1.00      1.00         1
              22       0.60      0.75      0.67         4
              23       1.00      1.00      1.00         4
              24       0.67      1.00      0.80         2
              25       1.00      0.50      0.67         2
              26       1.00      0.50      0.67         4
              27       0.00      0.00      0.00         3
              28       1.00      1.00      1.00         2
              29       0.06      1.00      0.11         1
              30       0.00      0.00      0.00         1
              31       0.00      0.00      0.00         1
              32       1.00      0.33      0.50         3
              33       1.00      1.00      1.00         2
              34       0.00      0.00      0.00         1
              35       1.00      1.00      1.00         1
              36       1.00      0.50      0.67         2
              37       1.00      0.67      0.80         3
              38       1.00      0.40      0.57         5
              39       1.00      0.50      0.67         4
    
        accuracy                           0.58       100
       macro avg       0.65      0.59      0.56       100
    weighted avg       0.73      0.58      0.58       100
    
    [[0 0 0 ... 0 0 0]
     [0 0 0 ... 0 0 0]
     [0 0 1 ... 0 0 0]
     ...
     [0 0 0 ... 2 0 0]
     [0 0 0 ... 0 2 0]
     [0 0 0 ... 0 0 2]]
    

    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\sklearn\metrics\_classification.py:1334: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\sklearn\metrics\_classification.py:1334: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\sklearn\metrics\_classification.py:1334: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    


```python
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
target_names = [str(element) for element in np.arange(40)+1]
prediction_titles = [title(y_pred_test, y_test, target_names, i)
                     for i in range(y_pred_test.shape[0])]
plot_gallery(X_test, height, width, prediction_titles, n_row=2, n_col=6)
plt.show()
```


    
![png](/assets/images/eigenfaces/eigenfaces-11.png)
    



```python

```


```python

```


```python
https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
```


```python
https://github.com/Manaliagarwal/Eigen-Faces-fetch_olivetti_faces-/blob/main/EigneFaces.ipynb
```
