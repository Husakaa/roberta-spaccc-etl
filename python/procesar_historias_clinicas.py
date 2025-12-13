"""
Pipeline ETL simplificado - Historias Clínicas SPACCC
Modelo: PlanTL-GOB-ES/roberta-base-biomedical-clinical-es
"""

import os
import re
import csv
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

INPUT_DIR = "./SPACCC/corpus"
OUTPUT_CSV = "./staging_historias.csv"
MODELO = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
MAX_ARCHIVOS = 20  # None para procesar todos (total 1000)

ESPECIALIDADES = [
    "Oncología", "Urología", "Cardiología", "Neurología", 
    "Neumología", "Digestivo", "Nefrología", "Traumatología",
    "Endocrinología", "Dermatología", "Ginecología", "Hematología"
]

DIAGNOSTICOS = [
    "Neoplasia maligna", "Neoplasia benigna", "Infección",
    "Inflamatorio", "Degenerativo", "Traumático", "Vascular"
]

TRATAMIENTOS = [
    "Quirúrgico", "Farmacológico", "Quimioterapia", 
    "Radioterapia", "Conservador", "Observación"
]

# ==============================================================================
# CARGAR MODELO
# ==============================================================================

print(f"Cargando modelo {MODELO}...")
tokenizer = AutoTokenizer.from_pretrained(MODELO)
model = AutoModel.from_pretrained(MODELO)
model.eval()
print("Modelo cargado correctamente")

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def obtener_embedding(texto, max_length=512):
    """Obtiene el embedding de un texto usando RoBERTa"""
    inputs = tokenizer(
        texto, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Usar el embedding del token [CLS] (posición 0)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding


def similitud_coseno(emb1, emb2):
    """Calcula similitud de coseno entre dos embeddings"""
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


def clasificar_texto(texto, categorias, embeddings_categorias):
    """Clasifica un texto en la categoría más similar"""
    emb_texto = obtener_embedding(texto)
    
    mejor_categoria = None
    mejor_score = -1
    
    for cat, emb_cat in zip(categorias, embeddings_categorias):
        score = similitud_coseno(emb_texto, emb_cat)
        if score > mejor_score:
            mejor_score = score
            mejor_categoria = cat
    
    return mejor_categoria, mejor_score


def extraer_edad(texto):
    """Extrae la edad del paciente"""
    patrones = [
        r'(?:paciente|varón|mujer)\s+de\s+(\d{1,3})\s*(?:años|a\.)',
        r'(\d{1,3})\s*(?:años|a\.)\s+de\s+edad',
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
    """Extrae el sexo del paciente"""
    texto_lower = texto.lower()
    if re.search(r'\b(varón|hombre|masculino)\b', texto_lower):
        return "Masculino"
    if re.search(r'\b(mujer|femenino|femenina)\b', texto_lower):
        return "Femenino"
    return None


def extraer_datos_oncologicos(texto):
    """Extrae biomarcadores oncológicos"""
    datos = {}
    
    # Estadio TNM
    match = re.search(r'(T[0-4][a-c]?\s*N[0-3][a-c]?\s*M[0-1])', texto, re.IGNORECASE)
    if match:
        datos['estadio_tnm'] = match.group(1)
    
    # Ki67
    match = re.search(r'ki[\-\s]?67[:\s]*(\d{1,3})\s*%', texto, re.IGNORECASE)
    if match:
        datos['ki67_porcentaje'] = float(match.group(1))
    
    # HER2
    match = re.search(r'HER[\-\s]?2[:\s]*(positivo|negativo|\+{1,3})', texto, re.IGNORECASE)
    if match:
        datos['her2_status'] = match.group(1)
    
    # Gleason (próstata)
    match = re.search(r'Gleason\s+(\d{1,2})', texto, re.IGNORECASE)
    if match:
        datos['grado_histologico'] = f"Gleason {match.group(1)}"
    
    return datos


# ==============================================================================
# PRE-CALCULAR EMBEDDINGS DE CATEGORÍAS
# ==============================================================================

print("Calculando embeddings de categorías...")
emb_especialidades = [obtener_embedding(f"Especialidad médica: {cat}") for cat in ESPECIALIDADES]
emb_diagnosticos = [obtener_embedding(f"Diagnóstico: {cat}") for cat in DIAGNOSTICOS]
emb_tratamientos = [obtener_embedding(f"Tratamiento: {cat}") for cat in TRATAMIENTOS]
print("Embeddings calculados")

# ==============================================================================
# PROCESAR DOCUMENTOS
# ==============================================================================

def procesar_documento(nombre_archivo, texto):
    """Procesa un documento y retorna un diccionario con los datos extraídos"""
    
    # Clasificar
    especialidad, conf_esp = clasificar_texto(texto, ESPECIALIDADES, emb_especialidades)
    diagnostico, conf_dx = clasificar_texto(texto, DIAGNOSTICOS, emb_diagnosticos)
    tratamiento, conf_tx = clasificar_texto(texto, TRATAMIENTOS, emb_tratamientos)
    
    # Extraer datos básicos
    edad = extraer_edad(texto)
    sexo = extraer_sexo(texto)
    
    # Confidence promedio
    confidence = round((conf_esp + conf_dx + conf_tx) / 3, 4)
    
    # Datos base
    resultado = {
        'nombre_archivo': nombre_archivo,
        'texto_original': texto.replace('\n', ' ').replace('\r', ' '),
        'especialidad': especialidad,
        'diagnostico': diagnostico,
        'tratamiento': tratamiento,
        'edad_paciente': edad,
        'sexo_paciente': sexo,
        'confidence_score': confidence,
        'tipo_tumor': None,
        'localizacion': None,
        'estadio_tnm': None,
        'grado_histologico': None,
        'ki67_porcentaje': None,
        'her2_status': None,
        'receptor_estrogeno': None,
        'receptor_progesterona': None,
        'metastasis': None,
        'tratamiento_onco': None,
        'fecha_procesamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Si es oncología, extraer datos adicionales
    if especialidad == "Oncología":
        datos_onco = extraer_datos_oncologicos(texto)
        resultado.update(datos_onco)
    
    return resultado


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================

def main():
    print("\n" + "="*60)
    print("PIPELINE ETL - HISTORIAS CLÍNICAS SPACCC")
    print("="*60)
    
    # Verificar directorio
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"ERROR: Directorio no encontrado: {INPUT_DIR}")
        print("Crea el directorio y coloca los archivos .txt del corpus")
        return
    
    # Cargar archivos
    archivos = list(input_path.glob("*.txt"))
    if MAX_ARCHIVOS:
        archivos = archivos[:MAX_ARCHIVOS]
    print(f"\nEncontrados {len(archivos)} archivos .txt")
    
    # Procesar
    resultados = []
    for archivo in tqdm(archivos, desc="Procesando"):
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                texto = f.read()
            resultado = procesar_documento(archivo.name, texto)
            resultados.append(resultado)
        except Exception as e:
            print(f"Error en {archivo.name}: {e}")
    
    # Exportar CSV
    columnas = [
        'nombre_archivo', 'texto_original', 'especialidad', 'diagnostico',
        'tratamiento', 'edad_paciente', 'sexo_paciente', 'confidence_score',
        'tipo_tumor', 'localizacion', 'estadio_tnm', 'grado_histologico',
        'ki67_porcentaje', 'her2_status', 'receptor_estrogeno',
        'receptor_progesterona', 'metastasis', 'tratamiento_onco',
        'fecha_procesamiento'
    ]
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columnas, delimiter=';')
        writer.writeheader()
        writer.writerows(resultados)
    
    print(f"\n✅ CSV exportado: {OUTPUT_CSV}")
    
    # Estadísticas
    print("\n" + "="*60)
    print("ESTADÍSTICAS")
    print("="*60)
    print(f"Total procesados: {len(resultados)}")
    
    # Contar por especialidad
    conteo_esp = {}
    for r in resultados:
        esp = r['especialidad']
        conteo_esp[esp] = conteo_esp.get(esp, 0) + 1
    
    print("\nPor especialidad:")
    for esp, count in sorted(conteo_esp.items(), key=lambda x: -x[1]):
        print(f"  {esp}: {count}")
    
    # Contar oncología
    onco = sum(1 for r in resultados if r['especialidad'] == 'Oncología')
    print(f"\nCasos oncológicos: {onco}")


if __name__ == "__main__":
    main()