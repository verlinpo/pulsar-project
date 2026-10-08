"""
curtosis.py

Curtosis espectral (SK) multiescala, calculada a partir del .fil (S1) y del
archivo S2 que genera convertidor.convert_and_write(..., s2_path=...).

Idea:
    Para un canal, sean P_1 ... P_M las potencias |X|^2 de M FFTs seguidas.
        S1 = sum(P_i)        S2 = sum(P_i^2)
        SK = (M+1)/(M-1) * (M*S2/S1^2 - 1)          (Nita y Gary 2010, con N = d = 1)
    - Ruido gaussiano (el cielo): P sigue una exponencial, E[P^2] = 2 E[P]^2,
      y la SK vale 1 en promedio.
    - Senal continua de amplitud fija (portadora, tono): P casi no varia y la
      SK baja hacia 0.
    - Senal intermitente (chispas, rafagas): P tiene picos y la SK sube sobre 1.
    Lo importante es que no mira cuanta energia hay, sino COMO se distribuye.

Multiescala:
    Cada fila del .fil suma group_chunks FFTs, asi que una fila sola da M = 4:
    muy poca estadistica (desviacion ~1). Pero S1 y S2 son sumas, asi que
    sumando R filas se obtiene la SK con M = R * group_chunks, para varios R
    a la vez y sin volver a leer el IQ:
        R chico  -> bloques cortos: pilla chispas de microsegundos
        R grande -> bloques largos: pilla senales que se prenden y apagan lento

Uso:
    python curtosis.py 400psr_01.fil
    python curtosis.py 400psr_01.fil --escalas 16,256,4096 --sigmas 3
"""
import argparse
import json
import numpy as np


def leer_fil(fil_path, nchans):
    """Abre los datos del .fil (float32) como memmap (filas, canales), saltando la cabecera."""
    with open(fil_path, "rb") as f:
        head = f.read(65536)
    fin = head.find(b"HEADER_END")
    if fin < 0:
        raise ValueError(f"{fil_path} no tiene HEADER_END: no parece un filterbank sigproc")
    return np.memmap(fil_path, dtype=np.float32, mode="r", offset=fin + len("HEADER_END")).reshape(-1, nchans)


def sk_estimador(s1, s2, M):
    """SK de cada (bloque, canal) a partir de las sumas S1 y S2 de M potencias."""
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (M + 1) / (M - 1) * (M * s2 / s1 ** 2 - 1)


def sk_desviacion(M):
    """Desviacion estandar exacta de la SK para ruido gaussiano (tiende a 2/sqrt(M))."""
    return np.sqrt(4.0 * M ** 2 / ((M - 1) * (M + 2) * (M + 3)))


def sk_umbrales(M, n_sigma=3.0):
    """
    Umbrales simetricos 1 +- n_sigma * desviacion.
    Aproximacion: la distribucion de la SK es asimetrica (cola mas larga hacia
    arriba) cuando M es chico, asi que con M ~ 64 hay algo mas de falsas
    alarmas altas de lo que dice la normal. Nita y Gary (2010) dan umbrales
    exactos con distribuciones de Pearson: es una mejora pendiente.
    """
    s = sk_desviacion(M)
    return max(0.0, 1.0 - n_sigma * s), 1.0 + n_sigma * s


def leer_meta(s2_path):
    with open(str(s2_path) + ".json") as f:
        return json.load(f)


def sk_multiescala(fil_path, s2_path, escalas=(16, 256, 4096), n_sigma=3.0):
    """
    Calcula la SK sumando R filas, para cada R en `escalas`.

    Devuelve {M: dict} con, para cada escala:
        filas_por_bloque, duracion_bloque_s, sk (bloques, canales),
        alta / baja (mascaras booleanas sobre los umbrales), umbrales
    Las filas que sobran al final (menos de R) no se usan en esa escala.
    """
    meta = leer_meta(s2_path)
    nchans, gc = meta["channels"], meta["group_chunks"]
    s1 = leer_fil(fil_path, nchans)
    s2 = np.load(s2_path, mmap_mode="r")
    if s1.shape != s2.shape:
        raise ValueError(f"El .fil tiene forma {s1.shape} y el S2 {s2.shape}: no son del mismo archivo")

    resultado = {}
    for R in escalas:
        nb = s1.shape[0] // R
        if nb == 0:
            continue
        S1 = s1[:nb * R].reshape(nb, R, nchans).sum(axis=1, dtype=np.float64)
        S2 = s2[:nb * R].reshape(nb, R, nchans).sum(axis=1, dtype=np.float64)
        M = R * gc
        sk = sk_estimador(S1, S2, M)
        lo, hi = sk_umbrales(M, n_sigma)
        resultado[M] = {
            "filas_por_bloque": R,
            "duracion_bloque_s": R * meta["tsamp"],
            "sk": sk,
            "alta": sk > hi,
            "baja": sk < lo,
            "umbrales": (lo, hi),
        }
    return resultado


def main():
    ap = argparse.ArgumentParser(description="Curtosis espectral multiescala de un .fil")
    ap.add_argument("fil", help="archivo .fil generado por test.py")
    ap.add_argument("--s2", default=None, help="archivo S2 (por defecto: mismo nombre con .s2.npy)")
    ap.add_argument("--escalas", default="16,256,4096", help="filas por bloque, separadas por coma")
    ap.add_argument("--sigmas", type=float, default=3.0, help="ancho de los umbrales en desviaciones")
    args = ap.parse_args()

    s2_path = args.s2 or args.fil.removesuffix(".fil") + ".s2.npy"
    escalas = [int(x) for x in args.escalas.split(",")]
    res = sk_multiescala(args.fil, s2_path, escalas, args.sigmas)

    print(f"{'M':>7} {'bloque':>10} {'umbral bajo':>12} {'umbral alto':>12} {'% alta':>8} {'% baja':>8}")
    for M, r in res.items():
        lo, hi = r["umbrales"]
        print(f"{M:>7} {r['duracion_bloque_s']*1e3:>8.2f}ms {lo:>12.3f} {hi:>12.3f} "
              f"{100*r['alta'].mean():>7.2f}% {100*r['baja'].mean():>7.2f}%")


if __name__ == "__main__":
    main()
