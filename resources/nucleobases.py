import numpy as np
from mne_features.univariate import *
from mne.epochs import BaseEpochs
np.seterr(divide='ignore', invalid='ignore')

def rms(data: np.ndarray) -> np.ndarray:
    values = compute_rms(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def mean(data: np.ndarray) -> np.ndarray:
    values = compute_mean(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def variance(data: np.ndarray) -> np.ndarray:
    values = compute_variance(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def skewness(data: np.ndarray) -> np.ndarray:
    values = compute_skewness(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def kurtosis(data: np.ndarray) -> np.ndarray:
    values = compute_kurtosis(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def std(data: np.ndarray) -> np.ndarray:
    values = compute_std(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def quantile(data: np.ndarray, q: float) -> np.ndarray:
    values = compute_quantile(data=data, q=q)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def ptp_amp(data: np.ndarray) -> np.ndarray:
    values = compute_ptp_amp(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def hurst_exp(data: np.ndarray) -> np.ndarray:
    values = compute_hurst_exp(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def hjorth_mobility(data: np.ndarray) -> np.ndarray:
    values = compute_hjorth_mobility(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def hjorth_complexity(data: np.ndarray) -> np.ndarray:
    values = compute_hjorth_complexity(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def higuchi_fd(data: np.ndarray, kmax: int) -> np.ndarray:
    values = compute_higuchi_fd(data=data.astype(np.float64), kmax=int(kmax))
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def katz_fd(data: np.ndarray) -> np.ndarray:
    values = compute_katz_fd(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def line_length(data: np.ndarray) -> np.ndarray:
    values = compute_line_length(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def app_entropy(data: np.ndarray, emb: int, metric: str) -> np.ndarray:
    values = compute_app_entropy(data=data, emb=emb, metric=metric)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def samp_entropy(data: np.ndarray, emb: int, metric: str) -> np.ndarray:
    try:
        values = compute_samp_entropy(data=data, emb=emb, metric=metric)
        if len(values) == 2:
            result = values[0]/values[1]
        else:
            result = values[0]
    except ValueError:
        result = 0
    return result

def decorr_time(data: np.ndarray, sfreq: float) -> np.ndarray:
    values = compute_decorr_time(data=data, sfreq=sfreq)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def pow_spect(data: np.ndarray, sfreq: float, frequency_bands: list, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'freq_bands': np.array(frequency_bands),
        'psd_method': psd_method, 
        'normalize': False, 
        'psd_params': None,
        'ratios': None,
        'log': False
    }
    values = compute_pow_freq_bands(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def pow_spect_log(data: np.ndarray, sfreq: float, frequency_bands: list, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'freq_bands': np.array(frequency_bands),
        'psd_method': psd_method, 
        'normalize': False, 
        'psd_params': None,
        'ratios': None,
        'log': True
    }
    values = compute_pow_freq_bands(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def pow_spect_ratios(data: np.ndarray, sfreq: float, frequency_bands: list, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'freq_bands': np.array(frequency_bands),
        'psd_method': psd_method, 
        'normalize': False, 
        'psd_params': None,
        'ratios_triu': True,
        'ratios': 'only',
        'log': False
    }
    values = compute_pow_freq_bands(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values
    return result

def pow_spect_log_ratios(data: np.ndarray, sfreq: float, frequency_bands: list, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'freq_bands': np.array(frequency_bands),
        'psd_method': psd_method, 
        'normalize': False, 
        'psd_params': None,
        'ratios_triu': True,
        'ratios': 'only',
        'log': True
    }
    values = compute_pow_freq_bands(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values
    return result

def hjorth_mobility_spect(data: np.ndarray, sfreq: float, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'psd_method': psd_method, 
        'normalize': False,
        'psd_params': None
    }
    values = compute_hjorth_mobility_spect(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def hjorth_complexity_spect(data: np.ndarray, sfreq: float, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'psd_method': psd_method, 
        'normalize': False,
        'psd_params': None
    }
    values = compute_hjorth_complexity_spect(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def zero_crossings(data: np.ndarray) -> np.ndarray:
    values = compute_zero_crossings(data=data)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def spect_entropy(data: np.ndarray, sfreq: float, psd_method: str) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq,
        'psd_method': psd_method,
        'psd_params': None
    }
    values = compute_spect_entropy(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def svd_entropy(data: np.ndarray, tau: int, emb: int) -> np.ndarray:
    values = compute_svd_entropy(data=data, tau=tau, emb=emb)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def svd_fisher_info(data: np.ndarray, tau: int, emb: int) -> np.ndarray:
    values = compute_svd_fisher_info(data=data, tau=tau, emb=emb)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def energy_freq_bands(data: np.ndarray, sfreq: float, frequency_bands: list, deriv_filt: bool) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'freq_bands': np.array(frequency_bands),
        'deriv_filt': deriv_filt, 
    }
    values = compute_energy_freq_bands(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result

def spect_edge_freq(data: np.ndarray, sfreq: float, psd_method: str, ref_freq: float, edge: float) -> np.ndarray:
    kwargs = {
        'data': data,
        'sfreq': sfreq, 
        'psd_method': psd_method,
        'ref_freq': sfreq//ref_freq,
        'edge': [edge],
        'psd_params': None 
    }
    values = compute_spect_edge_freq(**kwargs)
    if len(values) == 2:
        result = values[0]/values[1]
    else:
        result = values[0]
    return result