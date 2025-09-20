import h5py
import numpy as np
from scipy.linalg import eigh

# Lee el archivo eri_reiher.h5
with h5py.File("FeMoCo/integrals/eri_reiher.h5", "r") as f:
    eri = f["eri"][()]  # (norb, norb, norb, norb)

norb = eri.shape[0]
print(f"[INFO] Cargando ERI con {norb} orbitales → tamaño {eri.shape}")

# Reordena a forma matricial (N^2, N^2)
eri_matrix = eri.transpose(0, 2, 1, 3).reshape(norb**2, norb**2)

# Descomposición espectral (equivalente a Cholesky modificado)
eigvals, eigvecs = eigh(eri_matrix)

# Filtramos valores pequeños/negativos
threshold = 1e-10
idx = eigvals > threshold
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Construimos gvec, gval (formato típico de TFermion/QuantumChem)
gval = eigvals
gvec = eigvecs

print(f"[OK] Factorización completada con rank = {len(gval)}")

# Guardamos en eri_reiher_cholesky.h5
with h5py.File("FeMoCo/integrals/eri_reiher_cholesky.h5", "w") as f:
    f.create_dataset("gvec", data=gvec)
    f.create_dataset("gval", data=gval)

print("[OK] Escrito FeMoCo/integrals/eri_reiher_cholesky.h5")
