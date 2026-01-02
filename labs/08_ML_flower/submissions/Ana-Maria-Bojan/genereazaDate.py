import pandas as pd
import numpy as np
from pathlib import Path


OUTPUT_PATH = Path("data/work/Ana-Maria-Bojan/lab08/expression_matrix_Ana-Maria-Bojan.csv")

def generate_dataset():
    # Configurare
    n_samples = 150       # Număr total de probe (suficient pentru a evita eroarea de stratify)
    n_genes = 50          # Număr de gene (features)
    classes = ["Normal", "Cancer_A", "Cancer_B"] # 3 clase distincte
    
    # 1. Generăm zgomot de fond (valori random de expresie)
    # Media 100, deviație standard 20
    data = np.random.normal(loc=100, scale=20, size=(n_samples, n_genes))
    
    labels = []
    
    # 2. Atribuim clase și introducem "semnale" (pattern-uri)
    # Astfel încât algoritmii ML să aibă ce învăța
    for i in range(n_samples):
        # Alegem clasa ciclic (0, 1, 2, 0, 1, 2...)
        class_idx = i % len(classes)
        label = classes[class_idx]
        labels.append(label)
        
        # Modificăm datele pentru a crea diferențe reale între clase
        if label == "Normal":
            # Normal are primele 10 gene cu valori scăzute
            data[i, 0:10] -= 40
        elif label == "Cancer_A":
            # Cancer_A are genele 10-20 cu valori foarte mari
            data[i, 10:20] += 60
            # Și genele 40-45 scăzute
            data[i, 40:45] -= 30
        elif label == "Cancer_B":
            # Cancer_B are genele 20-30 mari și genele 0-5 mari
            data[i, 20:30] += 50
            data[i, 0:5] += 40

    # 3. Creăm DataFrame-ul
    # Numele coloanelor de gene: Gene_0, Gene_1 ...
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    df = pd.DataFrame(data, columns=gene_names)
    
    # Asigurăm că valorile sunt pozitive (expresia genică nu e negativă de obicei)
    df[df < 0] = 0
    
    # --- IMPORTANT ---
    # Adăugăm coloana Label LA FINAL (codul tău folosește iloc[:, -1] pentru label)
    df["Label"] = labels
    
    # 4. Salvare
    # Creăm folderul dacă nu există
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Dataset generat la: {OUTPUT_PATH}")
    print(f"Dimensiuni: {df.shape} (ultima coloană este Label)")
    print("Distribuția claselor:")
    print(df["Label"].value_counts())

if __name__ == "__main__":
    generate_dataset()