from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import io
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from itertools import cycle

%load_ext tensorboard

# 1. Carregar e preparar os dados (mantendo seu código original)
logdir = 'logs'

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 2. Definir o modelo (igual ao seu código original)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 3. Callbacks para TensorBoard (igual ao seu código original)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(logdir + '/cm')

def log_confusion_matrix(epoch, logs):
    test_pred = np.argmax(model.predict(test_images), axis=-1)
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=test_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    imagel = tf.image.decode_png(buf.getvalue(), channels=4)
    imagel = tf.expand_dims(imagel, 0)
    
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", imagel, step=epoch)

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

# 4. Compilar e treinar (igual ao seu código original)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=train_images, y=train_labels, epochs=5,
                   validation_data=(test_images, test_labels),
                   callbacks=[tensorboard_callback, cm_callback])

# 5. Avaliação completa do modelo (novas métricas adicionadas)
print("\n=== Avaliação Completa do Modelo ===")

# Obter previsões
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_labels

# 5.1 Classification Report
print("\n1. Classification Report:")
print(classification_report(y_true, y_pred_classes))

# 5.2 Métricas detalhadas por classe
def calculate_metrics(y_true, y_pred):
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    metrics = []
    
    for i in range(len(classes)):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        TN = cm.sum() - TP - FP - FN
        
        sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
        
        metrics.append({
            'Classe': i,
            'Sensibilidade (Recall)': sensitivity,
            'Especificidade': specificity,
            'Precisão': precision,
            'F1-Score': f1_score,
            'Acurácia': (TP + TN) / (TP + TN + FP + FN)
        })
    
    return pd.DataFrame(metrics)

metrics_df = calculate_metrics(y_true, y_pred_classes)
print("\n2. Métricas Detalhadas por Classe:")
print(metrics_df)

# 5.3 Acurácia Global
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\n3. Acurácia Global: {test_acc:.4f}")

# 5.4 Curva ROC Multiclasse
print("\n4. Gerando Curva ROC Multiclasse...")

# Binarizar as labels para formato one-hot
y_test_bin = label_binarize(y_true, classes=classes)
n_classes = y_test_bin.shape[1]

# Calcular ROC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotar todas as curvas ROC
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 
               'orange', 'pink', 'brown', 'gray', 'olive'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='Classe {0} (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC Multiclasse')
plt.legend(loc="lower right")
plt.show()

# 5.5 Matriz de Confusão Final (como no seu código original)
print("\n5. Matriz de Confusão Final:")
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred_classes).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.title('Matriz de Confusão Normalizada')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

%tensorboard --logdir logs
