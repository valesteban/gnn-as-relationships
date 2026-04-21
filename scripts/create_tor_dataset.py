import bz2
import csv
import numpy as np
from collections import Counter
# from utils import export_dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



TOR_LABELS_DICT = {'P2P':0, 'C2P': 1,'P2C': 2}
TOR_CSV_LABELS_DICT = {'P2P':0,'P2C': -1}
TEST_SIZE = 0.2

PATH = "/media/vale/KINGSTON/TESIS/"
DATA = "data_2026/"

# Dataset entrenamiento RNA para relaciones entre AS
ToR_CSV = 'CAIDA_AS_Relationships/Serial_2/20260301.as-rel2.txt.bz2'
ToR_CSV = 'CAIDA_AS_Relationships/Serial_1/20260301.as-rel.txt.bz2'

tor_dataset = []
tor_labels = []
MES = "marzo"

print("CRREAR DATASET\n")


with bz2.open(PATH + DATA + ToR_CSV, 'rt') as csv_file:
    reader = csv.reader(csv_file,delimiter='|')
    for i, row in enumerate(reader):
      if row[0][0] != '#' and int(row[2]) in TOR_CSV_LABELS_DICT.values():
        tor_dataset.append(np.asarray(row[:2]))
        tor_dataset.append(np.asarray(row[1::-1]))

      #   label_val = row[2]
      #   print(label_val)
        
        if row[2] == '-1':
           tor_labels.append(int(2))
           tor_labels.append(int(1))
        if row[2] == '0':
           tor_labels.append(int(0))
           tor_labels.append(int(0))

tor_dataset = np.array(tor_dataset)
tor_labels = np.array(tor_labels)


count = Counter(tor_labels)
print("[COUNTER] P2P -> 0", count[0])  
print("[COUNTER] C2P -> 1", count[1])
print("[COUNTER] P2C -> 2", count[2])      

# Exportar dataset---------------------------------------------------------------

# # Guardar en archivos .npy
# np.save(PATH + DATA + "caida_tor_dataset.npy", tor_dataset)
# np.save(PATH + DATA +"caida_tor_labels.npy", tor_labels)
# print(f"Archivos guardados en {DATA} ")

# SPLIT DATASET ------------------------------------------------------------------------------
print("SPLIT DATASET\n")


indices = np.arange(len(tor_dataset))
np.random.shuffle(indices)
tor_dataset = tor_dataset[indices]
tor_labels = tor_labels[indices]
# dataset, labels = shuffle(dataset, labels, random_state=7)


indices = np.arange(len(tor_dataset))
x_training, x_test, indices_training, indices_test = train_test_split(tor_dataset, indices, test_size=TEST_SIZE)
y_training = tor_labels[indices_training]
y_test = tor_labels[indices_test]

print("[Y trainnig] ----------------")
count = Counter(y_training)
print("[COUNTER] ", count)  
print("[COUNTER] P2P -> 0", count[0])  
print("[COUNTER] C2P -> 1", count[1])
print("[COUNTER] P2C -> 2", count[2])  

print("[Y test] ----------------")
count = Counter(y_test)
print("[COUNTER] ", count)  
print("[COUNTER] P2P -> 0", count[0])  
print("[COUNTER] C2P -> 1", count[1])
print("[COUNTER] P2C -> 2", count[2])  

# Guardar x_training y x_test
np.save(PATH + DATA + f"ToR_data/tor_data_{MES}/x_training.npy", x_training)
np.save(PATH + DATA + f"ToR_data/tor_data_{MES}/x_test.npy", x_test)

# Guardar y_training y y_test
np.save(PATH + DATA + f"ToR_data/tor_data_{MES}/y_training.npy", y_training)
np.save(PATH + DATA + f"ToR_data/tor_data_{MES}/y_test.npy", y_test)

print("Archivos guardados con éxito! ", PATH+DATA)