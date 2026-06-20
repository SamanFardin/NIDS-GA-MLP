import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
mask = np.load('best_genetic_features.npy').astype(bool)

#حذف ستون های با ارزش پایین تر(اعمال ماسک)
x_train_selected = X_train[:, mask == 1]
x_test_selected = X_test[:, mask == 1]

#ساخت مدل
mlp = MLPClassifier(
    hidden_layer_sizes=(75, 100, 150, 100), 
    max_iter=500, 
    random_state=42, 
    early_stopping=True,# های اضافهepoch  برای حلوگیری از
    n_iter_no_change=10
)

#آموزش مدل
mlp.fit(x_train_selected, Y_train)

#تست مدل
prediction = mlp.predict(x_test_selected)

accuracy = accuracy_score(Y_test, prediction)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

report = classification_report(Y_test, prediction)
print(report)