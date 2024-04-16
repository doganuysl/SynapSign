import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, average_precision_score, precision_score, ConfusionMatrixDisplay
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

# Eğitimli modeli kaydetme
f = open('model_xgboost.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
