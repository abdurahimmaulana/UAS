import numpy as np
from sklearn.ensamble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)
x = [[1,2,3], #2 samples(row) 3 features(columns)
     [11,12,13]]
y = [0,1] #classes of each sample(row)
clf.fit(x,y)
RandomForestClassifier(random_state=0)
#print(clf)
#y_c = np.array([26,37,43]).reshape((-1,1))
y_c = clf.predict(X)
#array([0,1])
x_c = clf.predict([[17,19,14],
                  [0,5,1,7,2,9]])
#array)[0,1])
print('new prediction:',x_c)
