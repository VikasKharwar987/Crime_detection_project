import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = np.load('features.npy') 
y = np.load('features_labels.npy')   

print("Data loaded:", X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    GRU(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training started...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

model.save('crime_detection_model.keras')
print("Model saved as 'crime_detection_model'")
