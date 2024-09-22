import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split

# get data
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# make sets
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# create NN
model = tf.keras.models.Sequential()

# add hidden layer (8 units), ReLU activation
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu"))

# add output layer (1 unit), sigmoid activation
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Train NN
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_training, y_training, epochs=20)

# Evaluate perfs
model.evaluate(X_testing, y_testing, verbose=2)
