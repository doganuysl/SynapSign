import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import numpy as np
TF_ENABLE_ONEDNN_OPTS=0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Etiketleri Label Encoding ile dönüştürme
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

x_train, x_test, y_train_encoded, y_test_encoded = train_test_split(data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z'}

# DNN Modeli Oluşturma
model = Sequential([
    Dense(128, input_shape=(max_length,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(labels_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# DNN Modelini Eğitme
model.fit(x_train, y_train_encoded, epochs=10, batch_size=8, validation_split=0.2)

# Test veri seti üzerinde tahmin yapma
y_predict = np.argmax(model.predict(x_test), axis=-1)

# Metrikleri hesaplama
cm = confusion_matrix(y_test_encoded, y_predict)
recall = recall_score(y_test_encoded, y_predict, average='macro', zero_division=0)
accuracy = accuracy_score(y_test_encoded, y_predict)
precision = precision_score(y_test_encoded, y_predict, average='macro')
f1 = f1_score(y_test_encoded, y_predict, average='macro')

print("Recall: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, F1 Score: {:.4f}".format(recall, accuracy, precision, f1))

# Confusion Matrix'i görselleştirme
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels_dict.values()))
disp.plot()
plt.show()

# Modeli kaydetme
model.save('model_dnn.keras')
