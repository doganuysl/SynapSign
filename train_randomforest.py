import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# Veri setinizdeki farklı uzunluktaki listeleri aynı boyutta listelere dönüştürme
max_length = max(len(item) for item in data_dict['data'])  # En uzun liste uzunluğunu al
data_processed = []

for item in data_dict['data']:
    # Eksik veriyi tamamlayarak, tüm listeleri aynı boyuta getirme
    item += [0] * (max_length - len(item))  # Örnek olarak eksik veriyi sıfırlarla tamamladık
    data_processed.append(item)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_randomforest.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
