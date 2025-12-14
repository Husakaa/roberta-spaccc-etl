"""
Pipeline ETL - Historias Clínicas SPACCC
Modelo: Zero-shot classification con batching para GPU
"""

import os
import re
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

INPUT_DIR = "./SPACCC/corpus"
OUTPUT_CSV = "./data/staging_historias.csv"
MAX_ARCHIVOS = None  # None para procesar todos
BATCH_SIZE = 16  # Ajustar según memoria GPU (8, 16, 32)

ESPECIALIDADES = [
    "Urología",           # Muy frecuente en SPACCC
    "Cirugía General",    # Casos quirúrgicos
    "Medicina Interna",   # Casos médicos generales
    "Traumatología",      # Fracturas, lesiones
    "Oncología",          # Tumores
    "Digestivo",          # Aparato digestivo
    "Cardiología",        # Casos cardíacos
    "Neurología",         # Sistema nervioso
    "Nefrología",         # Riñón
    "Neumología"          # Pulmón
]

DIAGNOSTICOS = [
    "Neoplasia maligna",          # Carcinomas, adenocarcinomas, cáncer
    "Neoplasia benigna",          # Tumores benignos, pólipos, adenomas
    "Quiste",                     # Quistes de cualquier tipo
    "Infección",                  # Bacteriana, viral, parasitaria
    "Traumatismo",                # Fracturas, heridas, lesiones
    "Enfermedad cardiovascular",  # Infarto, arritmias, insuficiencia cardíaca
    "Enfermedad neurológica",     # Parálisis, neuropatías, epilepsia
    "Patología inflamatoria",     # Inflamaciones, artritis
    "Enfermedad metabólica",      # Diabetes, tiroides, insuficiencia renal
    "Obstrucción",                # Litiasis, estenosis, hernias
    "Malformación congénita",     # Anomalías del desarrollo
    "Enfermedad autoinmune",      # Lupus, esclerosis, etc.
    "Patología vascular",         # Trombosis, aneurismas, varices
    "Lesión cutánea"              # Úlceras, dermatitis, linfedema
]

TRATAMIENTOS = [
    "Cirugía abierta",        # Intervención quirúrgica tradicional
    "Cirugía laparoscópica",  # Mínimamente invasiva
    "Tratamiento médico",     # Fármacos, antibióticos
    "Quimioterapia",          # Tratamiento oncológico
    "Radioterapia",           # Tratamiento oncológico
    "Endoscopia",             # Procedimientos endoscópicos
    "Observación",            # Seguimiento sin intervención
    "Combinado"               # Varios tratamientos
]

# ==============================================================================
# CARGAR MODELO
# ==============================================================================

print("Cargando modelo zero-shot classification en GPU...")
clasificador = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli",
    device=0,  # GPU
    batch_size=BATCH_SIZE
)
print("Modelo cargado en GPU")

# ==============================================================================
# FUNCIONES REGEX
# ==============================================================================

def extraer_edad(texto):
    patrones = [
        r'(?:paciente|varón|mujer|niño|niña|hombre)\s+de\s+(\d{1,3})\s*(?:años|a\.)',
        r'(\d{1,3})\s*(?:años|a\.)\s+de\s+edad',
        r'(\d{1,3})\s*años,?\s+(?:varón|mujer)',
    ]
    texto_lower = texto.lower()
    for patron in patrones:
        match = re.search(patron, texto_lower)
        if match:
            edad = int(match.group(1))
            if 0 < edad < 120:
                return edad
    return None


def extraer_sexo(texto):
    texto_lower = texto.lower()
    if re.search(r'\b(varón|hombre|masculino|niño)\b', texto_lower):
        return "Masculino"
    if re.search(r'\b(mujer|femenino|femenina|niña)\b', texto_lower):
        return "Femenino"
    return None


# ==============================================================================
# CLASIFICACIÓN EN BATCH
# ==============================================================================

def clasificar_batch(textos, etiquetas, hypothesis):
    """Clasifica una lista de textos en batch"""
    textos_truncados = [t for t in textos]
    
    resultados = clasificador(
        textos_truncados,
        candidate_labels=etiquetas,
        hypothesis_template=hypothesis
    )
    
    # Extraer mejor etiqueta y score de cada resultado
    return [(r['labels'][0], r['scores'][0]) for r in resultados]


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("PIPELINE ETL - HISTORIAS CLÍNICAS SPACCC ")
    print("="*60)
    
    # Cargar archivos
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"ERROR: Directorio no encontrado: {INPUT_DIR}")
        return
    
    archivos = list(input_path.glob("*.txt"))
    if MAX_ARCHIVOS:
        archivos = archivos[:MAX_ARCHIVOS]
    print(f"\nCargando {len(archivos)} archivos...")
    
    # Leer todos los textos
    nombres = []
    textos = []
    for archivo in tqdm(archivos, desc="Leyendo"):
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                textos.append(f.read())
                nombres.append(archivo.name)
        except Exception as e:
            print(f"Error leyendo {archivo.name}: {e}")
    
    print(f"\n Clasificando {len(textos)} documentos en batches de {BATCH_SIZE}...")
    
    # Clasificar en batches con progreso
    print("\n[1/3] Clasificando especialidades...")
    esp_results = []
    for i in tqdm(range(0, len(textos), BATCH_SIZE), desc="Especialidades"):
        batch = textos[i:i+BATCH_SIZE]
        esp_results.extend(clasificar_batch(batch, ESPECIALIDADES, "Este caso clínico es de {}."))
    
    print("\n[2/3] Clasificando diagnósticos...")
    dx_results = []
    for i in tqdm(range(0, len(textos), BATCH_SIZE), desc="Diagnósticos"):
        batch = textos[i:i+BATCH_SIZE]
        dx_results.extend(clasificar_batch(batch, DIAGNOSTICOS, "El diagnóstico principal es {}."))
    
    print("\n[3/3] Clasificando tratamientos...")
    tx_results = []
    for i in tqdm(range(0, len(textos), BATCH_SIZE), desc="Tratamientos"):
        batch = textos[i:i+BATCH_SIZE]
        tx_results.extend(clasificar_batch(batch, TRATAMIENTOS, "El tratamiento aplicado fue {}."))
    
    # Construir resultados
    print("\n Extrayendo datos demográficos...")
    resultados = []
    for i, (nombre, texto) in enumerate(zip(nombres, textos)):
        esp, conf_esp = esp_results[i]
        dx, conf_dx = dx_results[i]
        tx, conf_tx = tx_results[i]
        
        resultados.append({
            'nombre_archivo': nombre,
            'especialidad': esp,
            'diagnostico': dx,
            'tratamiento': tx,
            'edad_paciente': extraer_edad(texto) or '',
            'sexo_paciente': extraer_sexo(texto) or '',
            'confidence_score': round((conf_esp + conf_dx + conf_tx) / 3, 4),
            'fecha_procesamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Exportar CSV
    columnas = [
        'nombre_archivo', 'especialidad', 'diagnostico', 'tratamiento',
        'edad_paciente', 'sexo_paciente', 'confidence_score', 'fecha_procesamiento'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columnas, delimiter=';')
        writer.writeheader()
        writer.writerows(resultados)
    
    print(f"\n CSV exportado: {OUTPUT_CSV}")
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS")
    print("="*60)
    print(f"Total procesados: {len(resultados)}")
    
    for campo, nombre in [('especialidad', 'Especialidad'), ('diagnostico', 'Diagnóstico'), ('tratamiento', 'Tratamiento')]:
        conteo = {}
        for r in resultados:
            val = r[campo]
            conteo[val] = conteo.get(val, 0) + 1
        print(f"\nPor {nombre}:")
        for val, count in sorted(conteo.items(), key=lambda x: -x[1]):
            print(f"  {val}: {count}")


if __name__ == "__main__":
    main()