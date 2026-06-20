import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#نام دهی به ستون ها(دیتاست نام ستون ندارد)
data = pd.read_csv("KDDTest+.txt", header=None, names= [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]
)

#حذف ستون تا مدل از روی ان تقلب نکند
data.drop("difficulty_level", axis=1, inplace=True)

#تبدیل مقادیر ستون لیبل به 0 و1 
# 0 = normal 1= attack
data["label"] = data["label"].apply(lambda x: 0 if x == "normal" else 1) 

#One hot encoding colmun(protocol_type, service, flag)
data = pd.get_dummies(data, columns=["protocol_type", "service", "flag"])

#جداسازی ویژگی ها از ستون لیبل
target_label = data["label"].astype(np.int64)
Features = data.drop("label", axis=1)

#جداسازی داده های اموزش و تست
X_train, X_test, y_train, y_test = train_test_split(Features, target_label, test_size=0.2, random_state=42, stratify=target_label)

#اسکیل کردن فیچرها
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#ذخیره داده های پیش پردازش شده در هارد
np.save('X_train.npy', X_train)
np.save('Y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('Y_test.npy', y_test)

joblib.dump(scaler, "scaler.pkl")

print("Preprocessing done.")
print("Saved: X_train.npy, Y_train.npy, X_test.npy, Y_test.npy, scaler.pkl")