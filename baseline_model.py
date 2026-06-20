#Multi-layer Perceptron (MLP)

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# بارگذاری داده‌های پیش‌پردازش‌شده از هارد
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

#ساخت یک ابجکت از مدل
mlp = MLPClassifier(
    hidden_layer_sizes=(75, 100, 150, 100), 
    max_iter=500, 
    random_state=42
)

# Model Training
mlp.fit(X_train, Y_train)

# Model Test
prediction = mlp.predict(X_test)

accuracy = accuracy_score(Y_test, prediction)
print(f"Baseline Model Accuracy: {accuracy * 100:.2f}%")

report = classification_report(Y_test, prediction)
print(report)