"""
Uso:
    python test.py                          # usa la config de abajo
    python test.py 400psr_01.params         # otro .params
    python test.py 400psr_01.params 512      # forzar 512 canales
"""
import sys
from sampling_qol import load_data
from pulsar_main import write_header
from convertidor import convert_and_write
from optimizador_channels import optimal_channels, nearest_power_of_2, evaluate

PARAMS_FILE  = "400psr_01.params"
GROUP_CHUNKS = 4

# Canales: None = calcular automaticamente segun DM_SOURCE.
CHANNELS = None

DM_SOURCE = 67.77
PULSE_WIDTH_MS = 1.0

# Guardar S2 (suma de |X|^4) para la curtosis espectral: ocupa lo mismo que el .fil.
# Despues: python curtosis.py <archivo>.fil
GUARDAR_S2 = True

def main():
    params_file = sys.argv[1] if len(sys.argv) > 1 else PARAMS_FILE
    forced_channels = int(sys.argv[2]) if len(sys.argv) > 2 else CHANNELS

    obs = load_data(params_file)

    # --- decidir cuantos canales usar ---
    if forced_channels is not None:
        channels = forced_channels
        print(f"Canales fijados a mano: {channels}")
    else:
        n_opt = optimal_channels(DM_SOURCE, obs.sample_rate, obs.cfreq, GROUP_CHUNKS)
        channels = nearest_power_of_2(n_opt)
        print(f"Canales calculados automaticamente para DM={DM_SOURCE}:")
        print(f"  optimo teorico: {n_opt:.1f}  ->  usando {channels} ")

    # --- reportar que implica esa eleccion, antes de procesar ---
    m = evaluate(channels, DM_SOURCE, obs.sample_rate, obs.cfreq, GROUP_CHUNKS)
    print(f"\n  Ancho de canal    : {m['chan_width_MHz']:.4f} MHz")
    print(f"  Smearing intra-ch : {m['t_dm_ms']:.3f} ms")
    print(f"  Resolucion (tsamp): {m['tsamp_ms']:.3f} ms")
    print(f"  Ancho efectivo    : {m['effective_ms']:.3f} ms", end="")
    if m["effective_ms"] < PULSE_WIDTH_MS:
        print(f"  (< {PULSE_WIDTH_MS} ms del pulso: OK)")
    else:
        print(f"  (!) mayor que el pulso de {PULSE_WIDTH_MS} ms")
        print("      El pulso saldra ensanchado. Considerar bajar GROUP_CHUNKS.")

    # --- generar el .fil ---
    obs.set_channels(channels)

    outfil = str(obs.file).removesuffix(".iq").removesuffix(".bin") + ".fil"

    write_header(obs, group_chunks=GROUP_CHUNKS, outfile=outfil)

    s2_path = outfil.removesuffix(".fil") + ".s2.npy" if GUARDAR_S2 else None
    n_rows, tsamp = convert_and_write(
        obs.file, outfil,
        sample_rate=obs.sample_rate,
        channels=channels,
        group_chunks=GROUP_CHUNKS,
        dtype_bits=obs.sdr * 8,   # 8 para RTL-SDR, 16 para AirSpy
        s2_path=s2_path,
    )

    print(f"\nGenerado: {outfil}")
    if s2_path:
        print(f"  S2 para curtosis: {s2_path}")
    print(f"  {n_rows:,} filas x {channels} canales")
    print(f"  tsamp    = {tsamp*1e6:.1f} us")
    print(f"  Duracion = {n_rows*tsamp:.1f} s")

    verify_fil(outfil, obs, n_rows, tsamp)
    print(f"\nVerificar con: readfile {outfil}")


def verify_fil(outfil, obs, n_rows, tsamp):
    """Relee el .fil con sigpyproc (un lector independiente) y comprueba que
    la cabecera dice lo mismo que se calculo al convertir."""
    from sigpyproc.readers import FilReader

    hdr = FilReader(outfil).header
    checks = {
        "nchans": (hdr.nchans, obs.channels),
        "nsamples": (hdr.nsamples, n_rows),
        "tsamp": (hdr.tsamp, tsamp),
        "fch1": (hdr.fch1, obs.fch1),
        "foff": (hdr.foff, obs.foff),
    }
    ok = True
    for name, (got, expected) in checks.items():
        if abs(got - expected) > 1e-9 * max(1.0, abs(expected)):
            print(f"  (!) {name}: el archivo dice {got}, se esperaba {expected}")
            ok = False
    if ok:
        print(f"  Cabecera verificada: {hdr.nchans} canales de {hdr.fch1 + (hdr.nchans - 1) * hdr.foff:.4f}"
              f" a {hdr.fch1:.4f} MHz")

if __name__ == "__main__":
    main()