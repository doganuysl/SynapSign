import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


# Verileri yükleme
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Hedef değişkeni işleme
labels = np.asarray(data_dict['labels'])

# Label Encoding kullanarak string etiketleri sayısal hale dönüştürme
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Veri ön işleme (Liste uzunluklarını düzeltme)
max_len = max(len(item) for item in data_dict['data'])

data_processed = []
for item in data_dict['data']:
    item_padded = item + [0] * (max_len - len(item))
    data_processed.append(item_padded)

data = np.asarray(data_processed)

# Verileri eğitim ve test kümelerine ayırma
x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

# XGBoost sınıflandırıcı modeli oluşturma ve eğitme
model = XGBClassifier()
model.fit(x_train, y_train)

# Modelin tahminlerini yapma
y_predict = model.predict(x_test)

# Decoded Labels for Visualization (optional)
decoded_y_pred = le.inverse_transform(y_predict)
decoded_y_test = le.inverse_transform(y_test)

# Confusion Matrix with Decoded Labels (for visualization only)
cm_decoded = confusion_matrix(decoded_y_pred, decoded_y_test)

recall = recall_score(y_predict, y_test, average='macro', zero_division=0)
accuracy = accuracy_score(y_predict, y_test)
precision = precision_score(y_predict, y_test, average='macro')
f1 = f1_score(y_predict, y_test, average='macro')

print("Recall: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, F1 Score: {:.4f}".format(recall, accuracy, precision, f1))

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

disp = ConfusionMatrixDisplay(confusion_matrix=cm_decoded, display_labels=list(labels_dict.values()))
disp.plot()
plt.show()

# Eğitimli modeli kaydetme
f = open('model_xgboost.p', 'wb')
pickle.dump({'model': model}, f)
f.close()