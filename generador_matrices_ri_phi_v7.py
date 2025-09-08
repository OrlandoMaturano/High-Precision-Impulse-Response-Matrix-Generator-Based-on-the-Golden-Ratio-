# -*- coding: utf-8 -*-
"""
Generador de Matrices de Respuesta al Impulso 2x2 de Alta Precisión
basadas en la Proporción Áurea (φ).

Este script genera un conjunto de matrices de respuesta al impulso (RI) 2x2
implementando un modelo de síntesis de diafonía por reflexión simétrica.
Las distancias de reflexión se derivan de potencias de la proporción áurea (φ),
calculadas con una precisión de 10,000 decimales.

Versión 7 (Refactorización de Nomenclatura):
- Se actualiza toda la terminología del script para alinearse con
  estándares de la ingeniería de audio y el procesamiento de señales.
- Se mantienen intactas la lógica y la funcionalidad de la v6.

Entrada:
    h_LL.wav, h_LR.wav, h_RL.wav, h_RR.wav (impulsos canónicos)

Salida:
    13 paquetes ZIP (matriz_ri_phi_k.zip, ...)
    Cada ZIP contiene:
        - h_LL.wav, h_LR.wav, h_RL.wav, h_RR.wav (float de 64 bits)
        - metadata.json con el radio exacto, validaciones y SHA-256
"""

import os
import math
import json
import hashlib
import zipfile
import argparse
import logging
from decimal import Decimal, getcontext, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import soundfile as sf

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== PARÁMETROS FÍSICOS Y NUMÉRICOS =====================
SR = 48000
DEFAULT_TEMPERATURE = 20.0  # °C
DEFAULT_HUMIDITY = 50.0     # %
DEFAULT_PRESSURE = 101.325  # kPa
GAUSS_SIGMA = 0.0002        # 0.2 ms
AMP1 = 10**(-26/20)         # -26 dB
AMP2 = 10**(-32/20)         # -32 dB
TOL = 1e-15  # Tolerancia numérica
getcontext().prec = 11050   # Precisión para cálculos con Decimal

# --- Parámetros del Modelo de Síntesis de Diafonía ---
T1_BASE_MODELO = Decimal('0.0035')  # 3.5 ms
T2_BASE_MODELO = Decimal('0.0065')  # 6.5 ms
RATIO_RETARDO_REFLEXION = T2_BASE_MODELO / T1_BASE_MODELO

# Parámetros del filtro de retardo fraccionario
FRAC_DELAY_FILTER_LENGTH = 121
FRAC_DELAY_KAISER_BETA = 12.0

# ===================== FUNCIONES AUXILIARES DE DSP Y UTILIDADES =====================

def calculate_speed_of_sound(temperature: float = DEFAULT_TEMPERATURE, 
                           humidity: float = DEFAULT_HUMIDITY,
                           pressure: float = DEFAULT_PRESSURE) -> Decimal:
    """
    Calcula la velocidad del sonido precisa usando un modelo termodinámico (Cramer).
    """
    T = Decimal(temperature)
    RH = Decimal(humidity)
    P_kpa = Decimal(pressure)
    P = P_kpa * Decimal(10)  # Convertir kPa a hPa (milibares)

    # Presión de vapor saturado (fórmula de Magnus-Tetens, aproximación de Buck)
    P_sat = Decimal('6.112') * ((Decimal('17.67') * T) / (T + Decimal('243.5'))).exp()

    # Factor de mejora para aire húmedo
    f = Decimal('1.0016') + (Decimal('3.15e-6') * P) - (Decimal('0.074') / P)

    # Presión de vapor del agua en el aire
    P_v = (RH / Decimal(100)) * P_sat * f
    
    # Fracción molar de vapor de agua
    X_w = P_v / P
    
    # Constantes molares
    R_const = Decimal('8.314462')
    M_a = Decimal('28.9645e-3') # Masa molar del aire seco
    M_v = Decimal('18.01528e-3') # Masa molar del vapor de agua
    
    # Gamma (coeficiente de dilatación adiabática) para aire húmedo
    gamma = Decimal('1.4') * (Decimal(1) - Decimal('0.286') * X_w)
    
    # Masa molar del aire húmedo
    M_h = M_a * (Decimal(1) - X_w) + M_v * X_w
    
    # Temperatura en Kelvin
    T_k = T + Decimal('273.15')
    
    # Velocidad del sonido
    c = ( (gamma * R_const * T_k) / M_h ).sqrt()
    
    return c

def design_linear_phase_fractional_delay(delay_samples: float, 
                                         filter_length: int = FRAC_DELAY_FILTER_LENGTH) -> np.ndarray:
    """
    Diseña un filtro FIR de retardo fraccionario con fase lineal perfecta.
    Usa una ventana de Kaiser optimizada para minimizar el rizado.
    """
    if filter_length % 2 == 0:
        filter_length += 1
    
    n = np.arange(filter_length) - (filter_length - 1) / 2
    h = np.sinc(n - delay_samples)
    
    window = np.kaiser(filter_length, FRAC_DELAY_KAISER_BETA)
    
    h_windowed = h * window
    h_windowed /= np.sum(h_windowed)
    
    return h_windowed

def apply_linear_phase_fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    Aplica un retardo fraccionario de precisión mediante un filtro FIR de fase lineal.
    """
    delay_int = int(np.floor(delay_samples))
    delay_frac = delay_samples - delay_int
    
    if abs(delay_frac) < 1e-12: # Si el retardo es prácticamente entero
        delayed_signal = np.roll(signal, delay_int)
        if delay_int > 0: delayed_signal[:delay_int] = 0
        elif delay_int < 0: delayed_signal[delay_int:] = 0
        return delayed_signal
    
    fir_filter = design_linear_phase_fractional_delay(delay_frac)
    delayed_signal = np.convolve(signal, fir_filter, mode='same')
    
    delayed_signal = np.roll(delayed_signal, delay_int)
    if delay_int > 0: delayed_signal[:delay_int] = 0
    elif delay_int < 0: delayed_signal[delay_int:] = 0
    
    return delayed_signal

def enforce_perfect_symmetry(signal: np.ndarray) -> np.ndarray:
    """
    Fuerza simetría perfecta en una señal para eliminar errores numéricos.
    """
    if len(signal) % 2 == 0:
        mid = len(signal) // 2
        first_half = signal[:mid]
        second_half = signal[mid:]
        symmetric_second_half = first_half[::-1]
        signal[mid:] = (second_half + symmetric_second_half) / 2
        signal[:mid] = signal[mid:][::-1]
    else:
        mid = len(signal) // 2
        for i in range(1, mid + 1):
            avg = (signal[mid + i] + signal[mid - i]) / 2
            signal[mid + i] = avg
            signal[mid - i] = avg
    
    return signal

def _pad_signals_to_max_length(*signal_dicts: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Asegura que todos los arrays en los diccionarios de entrada tengan la misma longitud máxima."""
    max_len = 0
    for d in signal_dicts:
        for signal in d.values():
            max_len = max(max_len, len(signal))
    
    padded_dicts = []
    for d in signal_dicts:
        padded_d = {}
        for k, v in d.items():
            if len(v) < max_len:
                padded_d[k] = np.pad(v, (0, max_len - len(v)), 'constant')
            else:
                padded_d[k] = v.copy()
        padded_dicts.append(padded_d)
    return padded_dicts

def decimal_str(x: Decimal, n=10000) -> str:
    """Formatea un Decimal a string con n decimales redondeado correctamente."""
    q = Decimal(1).scaleb(-n)
    xq = x.quantize(q, rounding=ROUND_HALF_UP)
    s = f"{xq:f}"
    
    if "." not in s: s += "." + "0" * n
    else:
        ip, fp = s.split(".")
        s = ip + "." + fp.ljust(n, "0")[:n]
    
    return s

def sha256(filepath: str) -> str:
    """Calcula el hash SHA256 de un archivo para verificación de integridad."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

# ===================== FUNCIONES PRINCIPALES DE GENERACIÓN =====================

def cargar_impulsos_canonicos() -> Dict[str, np.ndarray]:
    """Carga los archivos WAV de impulso canónico y verifica sus propiedades."""
    archivos_impulso = {"h_LL": "h_LL.wav", "h_LR": "h_LR.wav", "h_RL": "h_RL.wav", "h_RR": "h_RR.wav"}
    impulsos_canonicos = {}
    for k, fname in archivos_impulso.items():
        if not os.path.exists(fname): raise FileNotFoundError(f"Archivo de impulso canónico no encontrado: {fname}")
        data, sr = sf.read(fname, dtype="float64")
        if sr != SR: raise ValueError(f"{fname} tiene SR {sr}, esperado {SR}")
        if data.ndim > 1: data = data[:, 0]
        impulsos_canonicos[k] = data
    
    padded_impulsos = _pad_signals_to_max_length(impulsos_canonicos)[0]
    return padded_impulsos

def radio_a_retardos(R: Decimal, speed_of_sound: Decimal) -> Tuple[Decimal, Decimal]:
    """Convierte un radio de reflexión a tiempos de retardo con precisión Decimal."""
    t1 = R / speed_of_sound
    t2 = t1 * RATIO_RETARDO_REFLEXION
    return (t1, t2)

def crear_tren_impulsos_gaussianos(longitud: int, retardos_s: List[Decimal], sigma_s: float, 
                                  amplitudes: List[float], indice_origen_t0: int, fs: int = SR) -> np.ndarray:
    """Crea un tren de impulsos gaussianos aplicando retardos fraccionarios de alta precisión."""
    tren_de_impulsos = np.zeros(longitud, dtype=np.float64)
    t = np.arange(longitud, dtype=np.float64) / fs
    tiempo_origen_s = indice_origen_t0 / fs
    fs_decimal = Decimal(fs)

    # Mapeo de retardos a amplitudes (t1 -> AMP1, t2 -> AMP2)
    map_retardo_a_amplitud = {
        abs(retardos_s[0]): amplitudes[0],
        abs(retardos_s[2]): amplitudes[1]
    }

    for retardo in retardos_s:
        amplitud = map_retardo_a_amplitud[abs(retardo)]
        
        retardo_en_muestras_decimal = retardo * fs_decimal
        retardo_en_muestras_float = float(retardo_en_muestras_decimal)

        gaussiana_base = amplitud * np.exp(-0.5 * ((t - tiempo_origen_s) ** 2) / (sigma_s ** 2))
        gaussiana_retardada = apply_linear_phase_fractional_delay(gaussiana_base, retardo_en_muestras_float)
        tren_de_impulsos += gaussiana_retardada
    
    return enforce_perfect_symmetry(tren_de_impulsos)

def sintetizar_reflexiones_simetricas(hLL_in: np.ndarray, hLR_in: np.ndarray, hRL_in: np.ndarray, hRR_in: np.ndarray,
                                     radio: Decimal, speed_of_sound: Decimal, indice_pico_impulso: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inyecta la señal de reflexión simétrica en la matriz de impulsos canónicos."""
    t1, t2 = radio_a_retardos(radio, speed_of_sound)

    max_retardo_float = float(max(t1, t2))
    longitud_necesaria = max(len(hLL_in), int(indice_pico_impulso + max_retardo_float * SR + 10 * GAUSS_SIGMA * SR))
    
    hLL_ext = np.pad(hLL_in, (0, longitud_necesaria - len(hLL_in)))
    hLR_ext = np.pad(hLR_in, (0, longitud_necesaria - len(hLR_in)))
    hRL_ext = np.pad(hRL_in, (0, longitud_necesaria - len(hRL_in)))
    hRR_ext = np.pad(hRR_in, (0, longitud_necesaria - len(hRR_in)))
    
    retardos_decimal = [t1, -t1, t2, -t2]
    amplitudes = [AMP1, AMP2]
    
    senal_de_reflexion = crear_tren_impulsos_gaussianos(longitud_necesaria, retardos_decimal, GAUSS_SIGMA, amplitudes, indice_pico_impulso)
    
    hLL_sintetizado = hLL_ext - senal_de_reflexion
    hLR_sintetizado = hLR_ext + senal_de_reflexion
    hRL_sintetizado = hRL_ext + senal_de_reflexion
    hRR_sintetizado = hRR_ext - senal_de_reflexion
    
    # Forzar simetría entre canales para robustez numérica
    hLL_sintetizado = enforce_perfect_symmetry(hLL_sintetizado)
    hLR_sintetizado = enforce_perfect_symmetry(hLR_sintetizado)
    hRL_sintetizado = enforce_perfect_symmetry(hRL_sintetizado)
    hRR_sintetizado = enforce_perfect_symmetry(hRR_sintetizado)
    
    return hLL_sintetizado, hLR_sintetizado, hRL_sintetizado, hRR_sintetizado

def apply_precise_headroom(signals: Dict[str, np.ndarray], headroom_db: float) -> Tuple[Dict[str, np.ndarray], float, float]:
    """Aplica headroom con precisión, manteniendo relaciones exactas entre canales."""
    pico_maximo = max(np.max(np.abs(signal)) for signal in signals.values())
    pico_objetivo = 10**(headroom_db/20)
    factor_escala = pico_objetivo / pico_maximo if pico_maximo > pico_objetivo else 1.0
    senales_escaladas = {key: signal * factor_escala for key, signal in signals.items()}
    return senales_escaladas, pico_maximo, factor_escala

def corregir_errores_de_identidad(senales_procesadas: Dict[str, np.ndarray], senales_originales: Dict[str, np.ndarray], factor_escala: float) -> Dict[str, np.ndarray]:
    """Corrige errores numéricos residuales para garantizar que las identidades de suma se cumplan perfectamente."""
    proc_padded, orig_padded = _pad_signals_to_max_length(senales_procesadas, senales_originales)
    
    # Corregir canal L: la suma de las salidas debe ser igual a la suma de las entradas escalada
    suma_L_deseada = (orig_padded["h_LL"] + orig_padded["h_LR"]) * factor_escala
    suma_L_actual = proc_padded["h_LL"] + proc_padded["h_LR"]
    error_L = suma_L_deseada - suma_L_actual
    proc_padded["h_LL"] += error_L / 2
    proc_padded["h_LR"] += error_L / 2
    
    # Corregir canal R
    suma_R_deseada = (orig_padded["h_RR"] + orig_padded["h_RL"]) * factor_escala
    suma_R_actual = proc_padded["h_RR"] + proc_padded["h_RL"]
    error_R = suma_R_deseada - suma_R_actual
    proc_padded["h_RR"] += error_R / 2
    proc_padded["h_RL"] += error_R / 2
    
    return proc_padded

def validar_identidades_de_suma(senales_procesadas: Dict[str, np.ndarray], senales_originales: Dict[str, np.ndarray], factor_escala: float) -> Tuple[float, float, float]:
    """Valida las identidades de suma y calcula el desequilibrio de nivel interaural (ILD) residual."""
    proc_padded, orig_padded = _pad_signals_to_max_length(senales_procesadas, senales_originales)

    suma_proc_L = proc_padded["h_LL"] + proc_padded["h_LR"]
    suma_proc_R = proc_padded["h_RR"] + proc_padded["h_RL"]
    suma_orig_L_escalada = (orig_padded["h_LL"] + orig_padded["h_LR"]) * factor_escala
    suma_orig_R_escalada = (orig_padded["h_RR"] + orig_padded["h_RL"]) * factor_escala
    
    error_L = float(np.max(np.abs(suma_proc_L - suma_orig_L_escalada)))
    error_R = float(np.max(np.abs(suma_proc_R - suma_orig_R_escalada)))
    
    Nfft = 1 << math.ceil(math.log2(len(suma_proc_L))) # Siguiente potencia de 2
    fft_L = np.fft.rfft(suma_proc_L, n=Nfft)
    fft_R = np.fft.rfft(suma_proc_R, n=Nfft)
    error_ild = float(np.max(np.abs(np.abs(fft_L) - np.abs(fft_R))))
    
    return error_L, error_R, error_ild

def generar_matriz_ri_para_radio(nombre: str, radio: Decimal, impulsos_canonicos_entrada: Dict[str, np.ndarray], 
                                 indice_pico_impulso: int, speed_of_sound: Decimal, output_dir: str, 
                                 headroom_db: float = -1) -> Dict:
    """Procesa un radio específico, generando la matriz de RI y sus metadatos de validación."""
    logging.info(f"Generando matriz de RI para: {nombre}")
    
    hLL_in, hLR_in, hRL_in, hRR_in = (
        impulsos_canonicos_entrada["h_LL"], impulsos_canonicos_entrada["h_LR"], 
        impulsos_canonicos_entrada["h_RL"], impulsos_canonicos_entrada["h_RR"]
    )
    
    hLL_sint, hLR_sint, hRL_sint, hRR_sint = sintetizar_reflexiones_simetricas(
        hLL_in.copy(), hLR_in.copy(), hRL_in.copy(), hRR_in.copy(), 
        radio, speed_of_sound, indice_pico_impulso
    )
    
    senales_sintetizadas = {"h_LL": hLL_sint, "h_LR": hLR_sint, "h_RL": hRL_sint, "h_RR": hRR_sint}
    senales_escaladas, pico_original, factor_escala = apply_precise_headroom(senales_sintetizadas, headroom_db)
    
    matriz_ri_final = corregir_errores_de_identidad(senales_escaladas, impulsos_canonicos_entrada, factor_escala)
    error_suma_L, error_suma_R, error_ild_residual = validar_identidades_de_suma(matriz_ri_final, impulsos_canonicos_entrada, factor_escala)
    
    os.makedirs(output_dir, exist_ok=True)
    
    archivos_wav_sha256 = {}
    for nombre_canal, arr in matriz_ri_final.items():
        nombre_archivo_wav = f"{nombre_canal}.wav"
        filepath = os.path.join(output_dir, nombre_archivo_wav)
        sf.write(filepath, arr.astype(np.float64), SR, subtype="DOUBLE")
        archivos_wav_sha256[nombre_archivo_wav] = sha256(filepath)
    
    radio_str = decimal_str(radio, 10000)
    t1, t2 = radio_a_retardos(radio, speed_of_sound)
    
    metadata = {
        "radio_reflexion_m": radio_str,
        "definicion_proporcion_aurea": "phi = (1 + sqrt(5)) / 2",
        "precision_aritmetica": "Python decimal, prec>=11050, ROUND_HALF_UP, 10 000 decimales",
        "velocidad_sonido_m_s": str(speed_of_sound),
        "condiciones_ambientales": {
            "temperatura_c": DEFAULT_TEMPERATURE, 
            "humedad_relativa_porc": DEFAULT_HUMIDITY, 
            "presion_atmosferica_kpa": DEFAULT_PRESSURE
        },
        "retardos_reflexion_s": {"t1": str(t1), "t2": str(t2), "ratio_t2_sobre_t1": decimal_str(RATIO_RETARDO_REFLEXION)},
        "amplitudes_reflexion_lineal": {"amp1_-26dB": AMP1, "amp2_-32dB": AMP2},
        "parametros_impulso_gaussiano": {
            "sigma_s": GAUSS_SIGMA, 
            "indice_pico_impulso_t0": indice_pico_impulso
        },
        "validaciones_de_precision": {
            "error_identidad_suma_L": error_suma_L, 
            "error_identidad_suma_R": error_suma_R, 
            "error_ild_residual_magnitud_fft": error_ild_residual,
            "pico_maximo_original": float(pico_original), 
            "pico_objetivo_normalizacion": float(10**(headroom_db/20)),
            "factor_escala_aplicado": factor_escala, 
            "tolerancia_numerica": TOL
        },
        "sha256_archivos_salida": archivos_wav_sha256,
        "parametros_filtro_retardo_fraccionario": {
            "metodo": "FIR de fase lineal con ventana Kaiser",
            "longitud_filtro": FRAC_DELAY_FILTER_LENGTH,
            "parametro_kaiser_beta": FRAC_DELAY_KAISER_BETA,
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    zipname = f"matriz_ri_{name}.zip"
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn_wav in archivos_wav_sha256:
            zf.write(os.path.join(output_dir, fn_wav), arcname=fn_wav)
        zf.write(metadata_path, arcname="metadata.json")
    
    if zipfile.ZipFile(zipname, "r").testzip() is not None:
        raise RuntimeError(f"El archivo ZIP {zipname} está corrupto")
    
    logging.info(f"Generado {zipname} - Error Suma L: {error_suma_L:.2e}, Error Suma R: {error_suma_R:.2e}, ILD Res: {error_ild_residual:.2e}")
    return metadata

def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Generador de Matrices de Respuesta al Impulso 2x2 basadas en la Proporción Áurea (φ)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperatura en °C (def: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--humidity", type=float, default=DEFAULT_HUMIDITY, help=f"Humedad relativa en % (def: {DEFAULT_HUMIDITY})")
    parser.add_argument("--pressure", type=float, default=DEFAULT_PRESSURE, help=f"Presión atmosférica en kPa (def: {DEFAULT_PRESSURE})")
    parser.add_argument("--headroom", type=float, default=-1, help="Headroom de normalización en dB (def: -1)")
    parser.add_argument("--output", type=str, default="output", help="Directorio de salida para los paquetes ZIP (def: 'output')")
    parser.add_argument("--verbose", action="store_true", help="Habilitar logging detallado")
    args = parser.parse_args()
    
    if args.verbose: logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        logging.info("Cargando impulsos canónicos de entrada...")
        impulsos_canonicos = cargar_impulsos_canonicos()
        
        indice_pico_impulso = int(np.argmax(np.abs(impulsos_canonicos["h_LL"] + impulsos_canonicos["h_LR"])))
        logging.info(f"Índice de pico de impulso (t=0) encontrado en la muestra: {indice_pico_impulso}")
        
        speed_of_sound = calculate_speed_of_sound(args.temperature, args.humidity, args.pressure)
        logging.info(f"Velocidad del sonido calculada: {speed_of_sound} m/s (a {args.temperature}°C, {args.humidity}% HR, {args.pressure} kPa)")
        
        phi = (Decimal(1) + Decimal(5).sqrt()) / Decimal(2)
        definiciones_de_radios = [
            ("phi_neg_5", phi**-5), ("phi_1", phi**1), ("phi_2", phi**2), ("phi_3", phi**3),
            ("phi_4", phi**4), ("phi_5", phi**5), ("phi_6", phi**6), ("phi_7", phi**7),
            ("phi_8", phi**8), ("phi_9", phi**9), ("phi_10", phi**10),
            ("phi_11_menos_phi_6", phi**11 - phi**6), ("phi_11", phi**11),
        ]
        
        os.makedirs(args.output, exist_ok=True)
        
        todos_los_metadatos = {}
        for nombre, radio in definiciones_de_radios:
            output_dir_individual = os.path.join(args.output, nombre)
            metadata = generar_matriz_ri_para_radio(
                nombre, radio, impulsos_canonicos, indice_pico_impulso, 
                speed_of_sound, output_dir_individual, args.headroom
            )
            todos_los_metadatos[nombre] = metadata
        
        logging.info("Procesamiento completado exitosamente.")
        print("\n--- Resumen de Errores de Precisión ---")
        for nombre, meta in todos_los_metadatos.items():
            v = meta["validaciones_de_precision"]
            logging.info(f"{nombre}: Error Suma L={v['error_identidad_suma_L']:.2e}, Error Suma R={v['error_identidad_suma_R']:.2e}, ILD Res={v['error_ild_residual_magnitud_fft']:.2e}")
        
    except Exception as e:
        logging.error(f"Se ha producido un error fatal durante el procesamiento: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()