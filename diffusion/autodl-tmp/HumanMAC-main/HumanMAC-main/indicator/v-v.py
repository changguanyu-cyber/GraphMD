import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# === Step 1: Load velocity data ===
velocities = np.load("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/velocities.npy")  # (T, N, 3)
T, N, _ = velocities.shape

# === Step 2: Flatten atomic velocities into a single velocity vector per frame ===
# Option 1: total center-of-mass velocity (less detailed)
# v_t = np.mean(velocities, axis=1)  # (T, 3)

# Option 2: all-atom velocity flattened (better for VACF)
v_t = velocities.reshape(T, -1)  # (T, N*3)

# === Step 3: Normalize each frame (zero mean optional) ===
v_t -= np.mean(v_t, axis=0)

# === Step 4: Compute VACF ===
vacf = np.zeros(T)
for tau in range(T):
    dot = np.sum(v_t[:T - tau] * v_t[tau:], axis=1)  # shape (T - tau,)
    vacf[tau] = np.mean(dot)

# === Step 5: FFT to get vibrational spectrum ===
dt_fs = 1.0  # Time step in fs, change this to your real value
dt_ps = dt_fs * 1e-3  # convert to ps

freqs = fftfreq(T, dt_ps)  # in THz
spectrum = np.abs(fft(vacf))

# Only take positive frequencies
positive = freqs > 0
freqs = freqs[positive]
spectrum = spectrum[positive]

# Optional: convert THz to cm^-1 (1 THz ≈ 33.356 cm⁻¹)
freqs_cm1 = freqs * 33.356

# === Step 6: Plot ===
plt.figure(figsize=(8, 5))
plt.plot(freqs_cm1, spectrum, lw=5)
plt.xlabel("Frequency (cm⁻¹)")
plt.ylabel("Intensity (a.u.)")
plt.title("Vibrational Spectrum from VACF")
plt.grid(True)
plt.xlim(0, 2500)
plt.tight_layout()
plt.savefig("/root/autodl-tmp/HumanMAC-main/HumanMAC-main/indicator/vibrational_spectrum.png", dpi=300)
plt.show()

