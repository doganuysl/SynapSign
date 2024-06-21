import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
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

cm = confusion_matrix(y_predict, y_test)
recall = recall_score(y_predict, y_test, average='macro', zero_division=0)
accuracy = accuracy_score(y_predict, y_test)
precision = precision_score(y_predict, y_test, average='macro')
f1 = f1_score(y_predict, y_test, average='macro')

print("Recall: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, F1 Score: {:.4f}".format(recall, accuracy, precision, f1))

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels_dict.values()))
disp.plot()
plt.show()


f = open('model_randomforest.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
