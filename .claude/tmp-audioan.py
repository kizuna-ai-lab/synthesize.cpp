#!/usr/bin/env python3
"""Objective characterisation of TTS output: is it speech or noise?

Usage: audioan.py <label>=<path.f32|path.wav> ...
"""
import sys, wave, struct, json
import numpy as np

SR = 24000

def read_wav_any(path):
    """Minimal RIFF reader handling PCM16 (fmt 1) and IEEE float32 (fmt 3)."""
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'RIFF' and data[8:12] == b'WAVE', path
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        cid = data[pos:pos+4]
        size = struct.unpack('<I', data[pos+4:pos+8])[0]
        body = data[pos+8:pos+8+size]
        if cid == b'fmt ':
            fmt = struct.unpack('<HHIIHH', body[:16])
        elif cid == b'data':
            pcm = body
        pos += 8 + size + (size & 1)
    tag, ch, sr, _, _, bits = fmt
    if tag == 1 and bits == 16:
        x = np.frombuffer(pcm, dtype='<i2').astype(np.float64) / 32768.0
    elif tag == 3 and bits == 32:
        x = np.frombuffer(pcm, dtype='<f4').astype(np.float64)
    else:
        raise ValueError(f'unsupported wav fmt tag={tag} bits={bits}')
    return x, ch


def load(path):
    if path.endswith('.f32'):
        x = np.fromfile(path, dtype='<f4')
    elif path.endswith('.wav'):
        x, ch = read_wav_any(path)
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
    else:
        raise ValueError(path)
    return x.astype(np.float64)


def frames(x, n=1024, hop=256):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, hop)
    return np.stack([x[i:i+n] for i in idx])


def analyse(x, sr=SR):
    out = {}
    out['samples'] = int(len(x))
    out['duration_s'] = round(len(x)/sr, 4)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    out['peak'] = round(peak, 6)
    out['rms'] = round(float(np.sqrt(np.mean(x**2))), 6)
    out['dc_offset'] = round(float(np.mean(x)), 6)

    F = frames(x)
    w = np.hanning(F.shape[1])
    Fw = F * w
    # frame energies
    fr_rms = np.sqrt(np.mean(F**2, axis=1))
    thr = max(peak * 0.02, 1e-6)
    out['silence_frac_frames_below_2pct_peak'] = round(float(np.mean(fr_rms < thr)), 4)

    # zero crossing rate (per second)
    zc = np.mean(np.abs(np.diff(np.signbit(F), axis=1)), axis=1)
    out['zcr_mean_per_sample'] = round(float(np.mean(zc)), 5)
    out['zcr_mean_hz'] = round(float(np.mean(zc)) * sr / 2, 1)

    # spectra
    S = np.abs(np.fft.rfft(Fw, axis=1))**2 + 1e-20
    freqs = np.fft.rfftfreq(F.shape[1], 1/sr)
    # restrict flatness to 50 Hz .. 8 kHz (speech band) to avoid DC/HF ringing dominance
    band = (freqs >= 50) & (freqs <= 8000)
    Sb = S[:, band]
    gm = np.exp(np.mean(np.log(Sb), axis=1))
    am = np.mean(Sb, axis=1)
    flat = gm/am
    # weight by frame energy so silence doesn't dominate
    wgt = fr_rms / (np.sum(fr_rms) + 1e-20)
    out['spectral_flatness_mean'] = round(float(np.mean(flat)), 4)
    out['spectral_flatness_energy_weighted'] = round(float(np.sum(flat*wgt)), 4)
    cent = np.sum(S*freqs[None, :], axis=1)/np.sum(S, axis=1)
    out['spectral_centroid_hz_energy_weighted'] = round(float(np.sum(cent*wgt)), 1)

    # band energy distribution (fraction of total power)
    edges = [0, 300, 1000, 3000, 6000, 9000, 12000]
    tot = np.sum(S, axis=1)
    bands = {}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (freqs >= a) & (freqs < b)
        bands[f'{a}-{b}Hz'] = round(float(np.sum((np.sum(S[:, m], axis=1)/tot)*wgt)), 4)
    out['band_power_fraction'] = bands

    # harmonicity: normalised autocorrelation peak in 60..400 Hz lag range, on voiced-ish frames
    lo, hi = int(sr/400), int(sr/60)
    hnr = []
    for f, e in zip(F, fr_rms):
        if e < thr:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, mode='full')[len(f)-1:]
        if ac[0] <= 0:
            continue
        ac = ac/ac[0]
        seg = ac[lo:hi]
        if len(seg) == 0:
            continue
        hnr.append(float(np.max(seg)))
    hnr = np.array(hnr) if hnr else np.array([0.0])
    out['n_active_frames'] = int(len(hnr))
    out['autocorr_peak_mean'] = round(float(np.mean(hnr)), 4)
    out['autocorr_peak_p90'] = round(float(np.percentile(hnr, 90)), 4)
    out['voiced_frac_autocorr_gt_0.4'] = round(float(np.mean(hnr > 0.4)), 4)
    out['voiced_frac_autocorr_gt_0.6'] = round(float(np.mean(hnr > 0.6)), 4)

    # spectral crest / peakiness of long-term average spectrum
    lts = np.mean(S, axis=0)
    out['lts_crest_db'] = round(float(10*np.log10(np.max(lts[band])/np.mean(lts[band]))), 2)
    # spectral flux (temporal variability) -- speech modulates, stationary noise doesn't
    Sn = np.sqrt(S)/ (np.sqrt(np.sum(S, axis=1, keepdims=True))+1e-20)
    flux = np.mean(np.abs(np.diff(Sn, axis=0)))
    out['spectral_flux_mean'] = round(float(flux*1e3), 4)
    # amplitude modulation: std/mean of frame rms over active frames
    act = fr_rms[fr_rms >= thr]
    if len(act) > 1:
        out['frame_rms_cv'] = round(float(np.std(act)/np.mean(act)), 4)
    else:
        out['frame_rms_cv'] = None
    return out


if __name__ == '__main__':
    res = {}
    for arg in sys.argv[1:]:
        label, path = arg.split('=', 1)
        res[label] = analyse(load(path))
    print(json.dumps(res, indent=2))
