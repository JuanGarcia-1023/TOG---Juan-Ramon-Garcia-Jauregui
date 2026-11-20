import os
import pandas as pd
import shutil
from sklearn.model_selection import GroupShuffleSplit

csv_path = r"C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\data\processed\archive\HAM10000_metadata.csv"
imagenes_originales_dir = r"C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\data\processed\archive\HAM10000"
directorio_salida = r"C:\Users\juanj\OneDrive\Documentos\ITESO\Cuarto Semestre\IDI IV\IDI III\idi_class_repo\data\processed\archive\Dataset_Binario" 

df = pd.read_csv(csv_path)

cancer_classes = ['mel', 'bcc', 'akiec']
df['clase_binaria'] = df['dx'].apply(lambda x: 'Cancer' if x in cancer_classes else 'No_Cancer')

gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
train_idx, temp_idx = next(gss1.split(df, groups=df['lesion_id']))

train_df = df.iloc[train_idx]
temp_df = df.iloc[temp_idx]

gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['lesion_id']))

val_df = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

def copiar_imagenes(dataframe, conjunto_nombre):
    for index, row in dataframe.iterrows():
        img_name = row['image_id'] + '.jpg'
        clase = row['clase_binaria']
        
        origen = os.path.join(imagenes_originales_dir, img_name)
        destino_dir = os.path.join(directorio_salida, conjunto_nombre, clase)
        destino = os.path.join(destino_dir, img_name)
        
        if not os.path.exists(destino_dir):
            os.makedirs(destino_dir)
            
        if os.path.exists(origen):
            shutil.copy2(origen, destino)
        else:
            print(f"Error {img_name} - No se encontró en origen")

print("Copiando Train (70%)...")
copiar_imagenes(train_df, 'Train')

print("Copiando Validation (15%)...")
copiar_imagenes(val_df, 'Validation')

print("Copiando Test (15%)...")
copiar_imagenes(test_df, 'Test')

print("Completado")
