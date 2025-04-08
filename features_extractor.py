import numpy as np
from scipy import signal
from scipy import stats
import pywt
import neurokit2 as nk

def process_ecg(signal, fs):
    """Procesa la señal ECG y detecta puntos característicos"""
    # Preprocesamiento
    signal_filtered = nk.ecg_clean(signal, sampling_rate=fs)

    # Detección robusta de picos R
    r_peaks = nk.ecg_findpeaks(signal_filtered, sampling_rate=fs)

    # Delineación de ondas
    try:
      signal, waves = nk.ecg_delineate(signal_filtered, r_peaks, sampling_rate=fs)
    except Exception as e:
      return None, None, None

    # Combinar información
    info = {
        'ECG_R_Peaks': r_peaks['ECG_R_Peaks'],
        'ECG_P_Peaks': waves['ECG_P_Peaks'],
        'ECG_Q_Peaks': waves['ECG_Q_Peaks'],
        'ECG_S_Peaks': waves['ECG_S_Peaks'],
        'ECG_T_Peaks': waves['ECG_T_Peaks'],
        'sampling_rate': fs
    }

    val = 1

    return signal_filtered, info, val


def extract_intervals(info):
    # Crear un diccionario para los intervalos y duraciones
    intervals_durations = {}

    # Frecuencia de muestreo
    fs = info['sampling_rate']

    # Extraer los picos de R, P, Q, S y T
    r_peaks = np.array(info['ECG_R_Peaks'])
    p_peaks = np.array(info['ECG_P_Peaks'])
    q_peaks = np.array(info['ECG_Q_Peaks'])
    s_peaks = np.array(info['ECG_S_Peaks'])
    t_peaks = np.array(info['ECG_T_Peaks'])

    # Intervalo RR (tiempo entre dos picos R consecutivos)
    rr_intervals = np.diff(r_peaks) / fs * 1000  # Convertir a milisegundos
    intervals_durations['RR_mean'] = np.nanmean(rr_intervals)  # Promedio de RR, excluyendo NaN

    # Intervalo PR (tiempo entre picos P y R)
    if len(p_peaks) > 0 and len(r_peaks) > 0:
        pr_intervals = (r_peaks[1:] - p_peaks[:-1]) / fs * 1000
        intervals_durations['PR_mean'] = np.nanmean(pr_intervals)  # Promedio de PR, excluyendo NaN

    # Duración QRS (tiempo entre picos Q y S)
    if len(q_peaks) > 0 and len(s_peaks) > 0:
        qrs_wide = 0
        qrs_durations = (s_peaks - q_peaks) / fs * 1000
        if np.any(qrs_durations > 120):
          #print("deteccion de qrs ancho, proabilidad de bloqueo en rama derecha")
          qrs_wide = np.sum(qrs_durations > 120) / len(qrs_durations)
        intervals_durations['QRS_Wide'] = qrs_wide
        intervals_durations['QRS_mean'] = np.nanmean(qrs_durations)  # Promedio de QRS, excluyendo NaN

    # Duración QT (tiempo entre picos Q y T)
    if len(q_peaks) > 0 and len(t_peaks) > 0:
        qt_durations = (t_peaks - q_peaks) / fs * 1000
        intervals_durations['QT_mean'] = np.nanmean(qt_durations)  # Promedio de QT, excluyendo NaN

    return intervals_durations

def extract_hrv(info, fs=500):
    hrv_data = nk.hrv_time(info["ECG_R_Peaks"], sampling_rate=fs).to_dict(orient='records')[0]
    hrv_filtered = {
      "HRV_SDNN": hrv_data.get("HRV_SDNN"),
      "HRV_RMSSD": hrv_data.get("HRV_RMSSD"),
      "HRV_pNN50": hrv_data.get("HRV_pNN50"),
      "HRV_CVNN": hrv_data.get("HRV_CVNN")
    }

    # Filtrar cualquier valor que sea mayor a 10 segundos (10000 ms)
    hrv_filtered = {key: value for key, value in hrv_filtered.items() if value is not None and value < 10000}

    return hrv_filtered

def extract_amplitudes(ecg_signal, info):
    # Obtener los picos detectados asegurando que no haya NaN
    peaks = np.array(info["ECG_R_Peaks"], dtype=int)

    # Manejar NaN en p_peaks y t_peaks
    p_peaks = np.array([int(x) for x in info.get("ECG_P_Peaks", []) if not np.isnan(x)], dtype=int)
    t_peaks = np.array([int(x) for x in info.get("ECG_T_Peaks", []) if not np.isnan(x)], dtype=int)
    q_peaks = np.array([int(x) for x in info.get("ECG_Q_Peaks", []) if not np.isnan(x)], dtype=int)
    s_peaks = np.array([int(x) for x in info.get("ECG_S_Peaks", []) if not np.isnan(x)], dtype=int)

    qrs_amplitudes = []
    for i in range(min(len(q_peaks), len(s_peaks))):
        qrs_amplitudes.append(ecg_signal[s_peaks[i]] - ecg_signal[q_peaks[i]])

    qrs_mean_amplitude = np.nanmean(qrs_amplitudes) if qrs_amplitudes else None

    # Asegurar que no estén vacíos antes de calcular la media
    def safe_mean(signal, indices):
        return np.mean(signal[indices]) if len(indices) > 0 else None

    return {
        "R_mean_amplitude": safe_mean(ecg_signal, peaks),
        "P_mean_amplitude": safe_mean(ecg_signal, p_peaks),
        "T_mean_amplitude": safe_mean(ecg_signal, t_peaks),
        "QRS_mean_amplitude": qrs_mean_amplitude,
    }


def extract_morphology_features(signal, info, fs):
    """Extrae características de la morfología de las ondas"""
    morphology = {}

    # Fragmentación QRS - importante para Chagas
    q_peaks = np.array([int(x) for x in info.get("ECG_Q_Peaks", []) if not np.isnan(x)], dtype=int)
    s_peaks = np.array([int(x) for x in info.get("ECG_S_Peaks", []) if not np.isnan(x)], dtype=int)

    if len(q_peaks) > 0 and len(s_peaks) > 0:
        # Extraer todos los complejos QRS
        qrs_complexes = []
        for i in range(min(len(q_peaks), len(s_peaks))):
            if q_peaks[i] < s_peaks[i]:
                qrs_complexes.append(signal[q_peaks[i]:s_peaks[i]])

        # Contar fragmentación (cambios de dirección en cada complejo QRS)
        frag_counts = []
        for qrs in qrs_complexes:
            if len(qrs) > 3:  # Asegurar que hay suficientes puntos
                # Contar cambios de dirección
                signs = np.sign(np.diff(qrs))
                changes = np.sum(np.abs(np.diff(signs))) / 2  # Dividir por 2 para contar solo cambios completos
                frag_counts.append(changes)

        morphology['QRS_fragmentation'] = np.mean(frag_counts) if frag_counts else np.nan

    # Características de la onda T (inversión, bifásica)
    t_peaks = np.array([int(x) for x in info.get("ECG_T_Peaks", []) if not np.isnan(x)], dtype=int)
    r_peaks = np.array([int(x) for x in info.get("ECG_R_Peaks", []) if not np.isnan(x)], dtype=int)

    if len(t_peaks) > 0 and len(r_peaks) > 0:
        t_amplitudes = signal[t_peaks]
        r_amplitudes = signal[r_peaks]

        # Relación T/R (característica importante en Chagas)
        morphology['T_R_ratio'] = np.mean(np.abs(t_amplitudes)) / np.mean(np.abs(r_amplitudes))

        # Detectar inversión de onda T
        morphology['T_inversion_ratio'] = np.sum(t_amplitudes < 0) / len(t_amplitudes)

    return morphology

def frequency_domain_analysis(ecg_signal, fs):
    f, Pxx = signal.welch(ecg_signal, fs=fs)

    # Calcular centroide espectral
    spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)

    return {
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': np.sqrt(np.sum(((f - spectral_centroid)**2 * Pxx)) / np.sum(Pxx)),
        'spectral_flatness': stats.gmean(Pxx) / np.mean(Pxx)
    }

def nonlinear_features(signal):
    # Entropía de muestra (Sample Entropy)
    def sample_entropy(time_series, m=2, r=0.2):
        def _chebyshev(x, y):
            return np.max(np.abs(x - y))

        def _count_matches(x, m, r):
            n = len(x)
            count_A, count_B = 0, 0

            for i in range(n - m):
                for j in range(n - m):
                    if i != j:
                        d_m = _chebyshev(x[i:i+m], x[j:j+m])
                        d_m1 = _chebyshev(x[i:i+m+1], x[j:j+m+1])

                        if d_m <= r:
                            count_A += 1
                        if d_m1 <= r:
                            count_B += 1

            return count_A, count_B

        A, B = _count_matches(time_series, m, r * np.std(time_series))
        return -np.log(A / B) if B > 0 else 0

    return {
        'sample_entropy': sample_entropy(signal),
        'largest_lyapunov_exponent': np.max(np.log(np.abs(np.diff(signal))))
    }

def additional_signal_characteristics(signal):
    return {
        'zero_crossings': np.sum(np.diff(np.signbit(signal))),
        'peak_to_peak_amplitude': np.max(signal) - np.min(signal),
        'signal_power': np.mean(signal**2)
    }

def complex_wavelet_transform(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)

    # Energía de cada nivel de descomposición
    energy_levels = [np.sum(np.abs(level)**2) for level in coeffs]

    return {f'wavelet_energy_level_{i}': energy for i, energy in enumerate(energy_levels)}

def extract_ecg_features(ecg_signal, channel=1, fs=400):
    ecg_channel = ecg_signal[:, channel]
    signal, info, val = process_ecg(ecg_channel, fs)

    if val == None:
      return None

    features = {
        "intervals": extract_intervals(info),
        "hrv": extract_hrv(info, fs),
        "morphology": extract_morphology_features(signal, info, fs),
        "amplitudes": extract_amplitudes(ecg_channel, info),
    }

    return features

def flatten_features_dict(features_dict):
    """Converts the nested feature dictionary into a 1D array."""
    flat_dict = {}
    for category, values in features_dict.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if key == 'sex_encoding':  # Handle sex_encoding separately
                    flat_dict[f"{category}_{key}"] = np.where(value)[0][0] if np.any(value) else 2
                else:
                    flat_dict[f"{category}_{key}"] = value
        else:
            flat_dict[category] = values

    feature_names = list(flat_dict.keys())
    feature_values = np.array([flat_dict[name] for name in feature_names])
    feature_values = np.nan_to_num(feature_values)

    return feature_values
