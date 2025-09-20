import os, sys
import h5py
import numpy as np
from pyscf.tools import fcidump
from pyscf.ao2mo import addons as ao2mo_addons

def main(fcidump_path, out_h5="integrals/eri_reiher.h5"):
    # 1) Leer FCIDUMP → dict con MAYÚSCULAS
    ctx = fcidump.read(fcidump_path, molpro_orbsym=False, verbose=False)
    # Claves esperadas por PySCF: 'H1','H2','ECORE','NORB','NELEC','MS2',...
    h1 = ctx['H1']                     # (norb, norb), simétrica
    h2_compact = ctx['H2']             # empaquetado (8-fold/4-fold), 1D o 2D
    n_orb = int(ctx['NORB'])
    n_elec = int(ctx['NELEC'])          # entero total (α+β)
    e_core = float(ctx.get('ECORE', 0.0))

    # 2) Restaurar H2 a 4 índices (pq|rs) en notación de químico
    #    - si es vector 1D (8-fold) o matriz 2D (4-fold), esto lo expande a 4D
    #    symmetry=1 → “sin simetría” (full 4D)
    h2 = ao2mo_addons.restore(1, h2_compact, n_orb)  # → shape (norb,norb,norb,norb)

    # 3) Crear carpeta integrals/ si no existe
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)

    # 4) Guardar en .h5 con nombres estándar y amigables
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("h0", data=h1)             # 1 cuerpo
        f.create_dataset("eri", data=h2)            # 2 cuerpos (pq|rs), 4D ya
        f.create_dataset("e_nuc", data=np.array(e_core))
        f.create_dataset("n_elec", data=np.array(n_elec))

    print(f"[OK] Escrito {out_h5}")
    print(f"     n_orb={n_orb}, n_elec={n_elec}, shapes: h1={h1.shape}, eri={h2.shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convert_to_h5.py /ruta/a/FCIDUMP [salida.h5]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "integrals/eri_reiher.h5"
    main(inp, out)



    with h5py.File("integrals/eri_reiher.h5", "r") as f:
        print(list(f.keys()))
        print("h1 shape:", f["h0"].shape)
        print("eri shape:", f["eri"].shape)
        print("n_elec:", f["n_elec"][()])
        print("e_nuc:", f["e_nuc"][()])
