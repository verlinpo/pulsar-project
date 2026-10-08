"""
conversion I/Q -> poder de canal que main_fixed.py (reconstruccion I/Q,
resta de DC offset, fftshift para orden ascendente de frecuencia), pero
vectorizada y procesada por batches.
La ganancia viene de:
    1. Leer el archivo UNA vez (np.memmap) en vez de abrirlo/buscarlo
       ~1.9 millones de veces.
    2. Hacer la FFT de un batch de miles de chunks de una sola llamada
       a np.fft.fft(..., axis=-1) en vez de un chunk a la vez.
    3. Trabajar en simple precision (float32/complex64) en vez de doble
       (float64/complex128): las muestras del SDR (uint8 o int16) entran
       exactas en la mantisa de 24 bits de float32, asi que no se pierde
       precision real.
    4. Calcular |z|**2 como z.real**2 + z.imag**2 en vez de abs(z)**2:
       abs() hace un sqrt elemento a elemento que el **2 de al lado
       inmediatamente deshace: es puro trabajo de sobra.
    5. convert_and_write() escribe cada batch a disco apenas se calcula,
       en vez de juntar el archivo completo en memoria (via
       convert_iq_to_filterbank_power) y recien ahi escribirlo.
Esto ya deja el conversor rapido en Python puro; la Tarea 7
(reescritura en C) seguiria siendo relevante solo si esto se sigue demorando.

Curtosis espectral (SK):
    convert_and_write(..., s2_path=...) guarda ademas S2 = suma de |X|^4 de
    cada fila, en un .npy aparte con la MISMA forma y orden de canales que el
    .fil (que ya contiene S1 = suma de |X|^2). El .fil no cambia en nada.
    Como S1 y S2 son sumas, se pueden sumar entre filas para calcular la SK a
    cualquier escala de tiempo sin volver a leer el IQ: ver curtosis.py.
"""
import json
import numpy as np


def _batch_geometry(n_complex_total, sample_rate, channels, group_chunks):
    samples_per_group = channels * group_chunks
    n_groups_total = n_complex_total // samples_per_group
    tsamp_integrated = samples_per_group / sample_rate
    return samples_per_group, n_groups_total, tsamp_integrated


def _iter_power_batches(raw, n_groups_total, samples_per_group, channels, group_chunks, dtype_bits, batch_groups,
                        with_s2=False):
    """
    Generador interno compartido: por cada batch hace la FFT + integracion
    de potencia y yield-ea (idx_inicio, S1 (g, channels), S2 (g, channels) o
    None) ya en orden sigproc (banda descendente). Usado tanto por
    convert_iq_to_filterbank_power (que junta todo en un array) como por
    convert_and_write (que escribe cada batch directo a disco).
    """
    dc_offset = 127.5 if dtype_bits == 8 else 0.0

    idx = 0
    complex_offset = 0
    while idx < n_groups_total:
        g = min(batch_groups, n_groups_total - idx)
        n_complex_batch = g * samples_per_group

        raw_batch = raw[complex_offset * 2: (complex_offset + n_complex_batch) * 2].astype(np.float32)
        if dc_offset:
            raw_batch -= dc_offset

        iq = np.empty(n_complex_batch, dtype=np.complex64)
        iq.real = raw_batch[0::2]
        iq.imag = raw_batch[1::2]

        blocks = iq.reshape(g, group_chunks, channels)
        fft_blocks = np.fft.fftshift(np.fft.fft(blocks, axis=2), axes=2)
        pow_blocks = fft_blocks.real ** 2 + fft_blocks.imag ** 2   # (g, group_chunks, channels)
        pow_desc = pow_blocks[:, :, ::-1]
        integrated = pow_desc.sum(axis=1, dtype=np.float32)
        # S2 se acumula en float64: |X|^4 llega a ~1e24 con int16 y 512 canales,
        # y al sumarlo en float32 se perderian los digitos que la SK necesita.
        s2 = np.square(pow_desc).sum(axis=1, dtype=np.float64).astype(np.float32) if with_s2 else None
        yield idx, integrated, s2
        idx += g
        complex_offset += n_complex_batch


def convert_iq_to_filterbank_power(
    input_path,
    sample_rate,
    channels=32,
    group_chunks=20,
    dtype_bits=8,
    batch_groups=5000,
):
    """
    Convierte un archivo I/Q crudo completo en un array 2D de poder por
    canal (tiempo x canal), vectorizada y por batches.
    """
    np_dtype = np.uint8 if dtype_bits == 8 else np.int16

    raw = np.memmap(input_path, dtype=np_dtype, mode="r")
    n_complex_total = raw.shape[0] // 2

    samples_per_group, n_groups_total, tsamp_integrated = _batch_geometry(
        n_complex_total, sample_rate, channels, group_chunks
    )

    power = np.empty((n_groups_total, channels), dtype=np.float32)
    for idx, integrated, _ in _iter_power_batches(
        raw, n_groups_total, samples_per_group, channels, group_chunks, dtype_bits, batch_groups
    ):
        power[idx:idx + integrated.shape[0]] = integrated

    return power, tsamp_integrated


def write_filterbank_body(power, outfile_path, mode="ab"):
    """
    Escribe el cuerpo del filterbank (los datos, sin header) a partir del
    array (n_groups, channels) devuelto por convert_iq_to_filterbank_power.
    """
    with open(outfile_path, mode) as fil:
        power.astype(np.float32, copy=False).tofile(fil)


def convert_and_write(
    input_path,
    outfile_path,
    sample_rate,
    channels=32,
    group_chunks=20,
    dtype_bits=8,
    batch_groups=5000,
    s2_path=None,
):
    """
    Escribe el cuerpo del .fil (S1 por fila). Si se da s2_path, guarda ademas
    S2 por fila en ese .npy y sus datos de lectura en s2_path + ".json".
    """
    np_dtype = np.uint8 if dtype_bits == 8 else np.int16

    raw = np.memmap(input_path, dtype=np_dtype, mode="r")
    n_complex_total = raw.shape[0] // 2

    samples_per_group, n_groups_total, tsamp_integrated = _batch_geometry(
        n_complex_total, sample_rate, channels, group_chunks
    )

    s2_out = None
    if s2_path is not None:
        # .npy guarda su propia forma y tipo, asi que se puede abrir con
        # np.load(..., mmap_mode="r") sin recordar cuantos canales tenia
        s2_out = np.lib.format.open_memmap(
            s2_path, mode="w+", dtype=np.float32, shape=(n_groups_total, channels)
        )

    n_rows_written = 0
    with open(outfile_path, "ab") as fil:
        for idx, integrated, s2 in _iter_power_batches(
            raw, n_groups_total, samples_per_group, channels, group_chunks, dtype_bits, batch_groups,
            with_s2=s2_out is not None,
        ):
            integrated.tofile(fil)
            if s2_out is not None:
                s2_out[idx:idx + integrated.shape[0]] = s2
            n_rows_written += integrated.shape[0]

    if s2_out is not None:
        s2_out.flush()
        del s2_out
        with open(str(s2_path) + ".json", "w") as meta:
            json.dump({
                "fil": str(outfile_path),
                "channels": channels,
                "group_chunks": group_chunks,   # espectros |X|^2 sumados en cada fila
                "sample_rate": sample_rate,
                "tsamp": tsamp_integrated,
                "n_rows": n_rows_written,
                "orden_canales": "descendente, igual que el .fil",
            }, meta, indent=2)

    return n_rows_written, tsamp_integrated
