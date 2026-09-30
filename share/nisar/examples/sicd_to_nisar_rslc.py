#!/usr/bin/env python3
"""
Convert a SICD (NITF 2.1) whose image grid is RGZERO / RMA-INCA into a NISAR RSLC.

Scope
-----
Only the SICD products whose image grid already is a zero-Doppler slant-range
grid: ``Grid/Type = RGZERO`` with ``RMA/ImageType = INCA`` and
``Grid/ImagePlane = SLANT`` (for example Capella stripmap). Polar-format
products (``Grid/Type = RGAZIM``) are rejected: they need new geometry, not a
repackaging.

The output follows the layout written by ``alos2_to_nisar_l1.py`` in this
directory, so that the isce3 RSLC readers and the InSAR and GCOV workflows can
open it.

Conventions
-----------
* Azimuth axis. For an INCA grid ``Grid/Col/UVectECF`` points along
  ``-look * velocity``, so zero-Doppler time decreases with column index on a
  left-looking collect. A NISAR RSLC needs it increasing. The flip is decided
  by the sign of ``RMA/INCA/TimeCAPoly[1]`` and applied to the image, the valid
  samples and the start and end times together.
* Pixels. SICD is range-major (rows = range, columns = azimuth); a NISAR RSLC
  is azimuth-major. The transpose is done in blocks of output lines.
* Slant range and time. Row ``r`` is at slant range
  ``R_CA_SCP + (r - SCPPixel.Row) * Grid/Row/SS`` and column ``c`` at
  closest-approach time ``TimeCAPoly((c - SCPPixel.Col) * Grid/Col/SS)``
  after ``Timeline/CollectStart``, which keeps its nanoseconds.
* Native Doppler. Zero by construction for an RGZERO grid; the LUT written
  under ``processingInformation/parameters`` is identically zero.
* Reference epoch. Midnight UTC of ``Timeline/CollectStart``.
* Radiometry (``--radiometry``). ``dn`` keeps the SICD digital numbers.
  ``beta0`` multiplies every pixel by ``sqrt(Radiometric/BetaZeroSFPoly)``
  evaluated at its own image coordinates (phase untouched), writes the noise
  LUT in beta0 power and adds ``calibrationInformation/geometry/beta0 = 1``,
  which is what ``nisar.workflows.gcov`` expects with
  ``input_terrain_radiometry: beta0``. In both modes the noise LUT evaluates
  ``Radiometric/NoiseLevel/NoisePoly`` (ABSOLUTE only), held at the nearest
  image edge outside the image.
* Valid samples. ``ImageData/ValidData`` is not used; the valid range interval
  of each line is derived from the data (first and last non-zero sample).

The verbatim SICD XML is stored in
``metadata/processingInformation/inputs/sicdXml``.

Only a single uncompressed image segment with 16-bit integer or 32-bit float
I/Q pixels is supported.

Example
-------
    sicd_to_nisar_rslc.py CAPELLA_..._SICD_HH_....ntf rslc.h5
    sicd_to_nisar_rslc.py CAPELLA_..._SICD_HH_....ntf rslc_beta0.h5 --radiometry beta0
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np

import isce3

SPEED_OF_LIGHT = 299792458.0
LSAR = "science/LSAR"
IDENT = f"{LSAR}/identification"
RSLC = f"{LSAR}/RSLC"
SWATHS = f"{RSLC}/swaths"
META = f"{RSLC}/metadata"


class UnsupportedSicdError(ValueError):
    """The SICD is valid but outside what this converter handles."""


# --------------------------------------------------------------------------
# NITF 2.1 container (MIL-STD-2500C)
# --------------------------------------------------------------------------
# File header fields up to HL (table A-1); every field is fixed-width ASCII.
_NITF_FILE_HEADER = [
    ("FHDR", 4), ("FVER", 5), ("CLEVEL", 2), ("STYPE", 4), ("OSTAID", 10),
    ("FDT", 14), ("FTITLE", 80),
    ("FSCLAS", 1), ("FSCLSY", 2), ("FSCODE", 11), ("FSCTLH", 2), ("FSREL", 20),
    ("FSDCTP", 2), ("FSDCDT", 8), ("FSDCXM", 4), ("FSDG", 1), ("FSDGDT", 8),
    ("FSCLTX", 43), ("FSCATP", 1), ("FSCAUT", 40), ("FSCRSN", 1),
    ("FSSRDT", 8), ("FSCTLN", 15),
    ("FSCOP", 5), ("FSCPYS", 5), ("ENCRYP", 1), ("FBKGC", 3),
    ("ONAME", 24), ("OPHONE", 18), ("FL", 12), ("HL", 6),
]

# Segment count fields and the widths of the (subheader, data) lengths that
# follow each count, in file order.
_NITF_SEGMENT_COUNTS = [
    ("NUMI", 6, 10),   # image segments
    ("NUMS", 4, 6),    # graphic segments
    ("NUMX", 0, 0),    # reserved
    ("NUMT", 4, 5),    # text segments
    ("NUMDES", 4, 9),  # data extension segments
]

# Image subheader fields up to ICORDS (table A-3).
_NITF_IMAGE_SUBHEADER = [
    ("IM", 2), ("IID1", 10), ("IDATIM", 14), ("TGTID", 17), ("IID2", 80),
    ("ISCLAS", 1), ("ISCLSY", 2), ("ISCODE", 11), ("ISCTLH", 2), ("ISREL", 20),
    ("ISDCTP", 2), ("ISDCDT", 8), ("ISDCXM", 4), ("ISDG", 1), ("ISDGDT", 8),
    ("ISCLTX", 43), ("ISCATP", 1), ("ISCAUT", 40), ("ISCRSN", 1),
    ("ISSRDT", 8), ("ISCTLN", 15), ("ENCRYP", 1), ("ISORCE", 42),
    ("NROWS", 8), ("NCOLS", 8), ("PVTYPE", 3), ("IREP", 8), ("ICAT", 8),
    ("ABPP", 2), ("PJUST", 1), ("ICORDS", 1),
]


def parse_nitf_header(head):
    """
    Parse a NITF 2.1 file header far enough to locate every segment.

    Parameters
    ----------
    head : bytes
        The first bytes of the file (2 KiB is enough for a SICD).

    Returns
    -------
    dict
        Header fields by name, plus ``"_segments"``: for each count field
        (``NUMI``, ``NUMDES``, ...) a list of dicts with ``subheader_offset``,
        ``subheader_length``, ``data_offset`` and ``data_length``.
    """
    if head[:4] != b"NITF":
        raise ValueError(f"not a NITF file (magic {head[:4]!r})")
    pos = 0
    fields = {}
    for name, width in _NITF_FILE_HEADER:
        fields[name] = head[pos:pos + width].decode("ascii", "replace")
        pos += width
    segments = {}
    cursor = int(fields["HL"])
    for count_name, sub_width, data_width in _NITF_SEGMENT_COUNTS:
        count = int(head[pos:pos + 3])
        pos += 3
        entries = []
        for _ in range(count):
            sub_len = int(head[pos:pos + sub_width]) if sub_width else 0
            pos += sub_width
            data_len = int(head[pos:pos + data_width]) if data_width else 0
            pos += data_width
            entries.append({"subheader_offset": cursor,
                            "subheader_length": sub_len,
                            "data_offset": cursor + sub_len,
                            "data_length": data_len})
            cursor += sub_len + data_len
        segments[count_name] = entries
    fields["_segments"] = segments
    return fields


def read_sicd_xml(path):
    """
    Return the SICD XML text stored in the first XML data extension segment.

    Parameters
    ----------
    path : str
        Path to the NITF file.

    Returns
    -------
    str
        The XML document.
    """
    with open(path, "rb") as fh:
        fields = parse_nitf_header(fh.read(2048))
        for des in fields["_segments"]["NUMDES"]:
            fh.seek(des["subheader_offset"])
            sub = fh.read(min(des["subheader_length"], 200))
            # The DES subheader begins with DE (2) + DESID (25).
            if "XML" not in sub[2:27].decode("ascii", "replace").upper():
                continue
            fh.seek(des["data_offset"])
            text = fh.read(des["data_length"]).decode("utf-8", "replace")
            start = text.find("<")
            if start >= 0:
                return text[start:]
    raise ValueError(f"{path}: no XML data extension segment")


def parse_image_subheader(raw):
    """
    Parse the NITF image subheader fields that decide the pixel layout.

    Parameters
    ----------
    raw : bytes
        The image subheader.

    Returns
    -------
    dict
        Subheader fields by name, with ``bands`` as a list of dicts.
    """
    pos = 0

    def take(n):
        nonlocal pos
        value = raw[pos:pos + n]
        pos += n
        return value

    d = {}
    for name, width in _NITF_IMAGE_SUBHEADER:
        d[name] = take(width).decode("latin1")
    if d["ICORDS"].strip():
        d["IGEOLO"] = take(60).decode("latin1")
    nicom = int(take(1))
    d["ICOM"] = [take(80).decode("latin1") for _ in range(nicom)]
    d["IC"] = take(2).decode("latin1")
    if d["IC"] not in ("NC", "NM"):
        d["COMRAT"] = take(4).decode("latin1")
    nbands = int(take(1))
    if nbands == 0:
        nbands = int(take(5))
    bands = []
    for _ in range(nbands):
        band = {"IREPBAND": take(2).decode("latin1"),
                "ISUBCAT": take(6).decode("latin1").strip(),
                "IFC": take(1).decode("latin1"),
                "IMFLT": take(3).decode("latin1")}
        nluts = int(take(1))
        if nluts > 0:
            nelut = int(take(5))
            take(nelut * nluts)
        bands.append(band)
    d["bands"] = bands
    d["ISYNC"] = take(1).decode("latin1")
    d["IMODE"] = take(1).decode("latin1")
    d["NBPR"] = int(take(4))
    d["NBPC"] = int(take(4))
    d["NPPBH"] = int(take(4))
    d["NPPBV"] = int(take(4))
    d["NBPP"] = int(take(2))
    d["NROWS"] = int(d["NROWS"])
    d["NCOLS"] = int(d["NCOLS"])
    d["ABPP"] = int(d["ABPP"])
    return d


def open_sicd_pixels(path, n_rg, n_az, pixel_type):
    """
    Memory-map the single SICD image segment after checking its layout.

    Parameters
    ----------
    path : str
        Path to the NITF file.
    n_rg, n_az : int
        ``ImageData/NumRows`` and ``ImageData/NumCols``.
    pixel_type : str
        ``ImageData/PixelType``.

    Returns
    -------
    mm : numpy.memmap
        Shape ``(n_rg, n_az, 2)`` in the file's big-endian type; the last axis
        is (I, Q).
    layout : dict
        Offsets and subheader fields, recorded in the output for provenance.
    """
    with open(path, "rb") as fh:
        fields = parse_nitf_header(fh.read(2048))
        images = fields["_segments"]["NUMI"]
        if len(images) != 1:
            raise UnsupportedSicdError(
                f"{len(images)} image segments; only a single-segment SICD is supported")
        seg = images[0]
        fh.seek(seg["subheader_offset"])
        sub = parse_image_subheader(fh.read(seg["subheader_length"]))

    expected = {"RE16I_IM16I": (">i2", "SI", 16), "RE32F_IM32F": (">f4", "R", 32)}
    if pixel_type not in expected:
        raise UnsupportedSicdError(f"ImageData/PixelType {pixel_type} is not supported")
    dtype, pvtype, nbpp = expected[pixel_type]

    problems = []
    if sub["IC"] != "NC":
        problems.append(f"IC={sub['IC']} (compressed)")
    if sub["IMODE"] != "P":
        problems.append(f"IMODE={sub['IMODE']} (need pixel-interleaved P)")
    if (sub["NBPR"], sub["NBPC"]) != (1, 1):
        problems.append(f"NBPR x NBPC = {sub['NBPR']} x {sub['NBPC']} (need 1 x 1)")
    if [b["ISUBCAT"] for b in sub["bands"]] != ["I", "Q"]:
        problems.append(f"bands {[b['ISUBCAT'] for b in sub['bands']]} (need I, Q)")
    if sub["PVTYPE"].strip() != pvtype or sub["NBPP"] != nbpp:
        problems.append(f"PVTYPE/NBPP {sub['PVTYPE']}/{sub['NBPP']} vs XML {pixel_type}")
    if (sub["NROWS"], sub["NCOLS"]) != (n_rg, n_az):
        problems.append(f"NROWS x NCOLS {sub['NROWS']} x {sub['NCOLS']} "
                        f"vs XML {n_rg} x {n_az}")
    nbytes = n_rg * n_az * 2 * (nbpp // 8)
    if seg["data_length"] != nbytes:
        problems.append(f"image segment length {seg['data_length']} != {nbytes}")
    if problems:
        raise UnsupportedSicdError("NITF image segment layout not supported: "
                                   + "; ".join(problems))

    mm = np.memmap(path, dtype=dtype, mode="r", offset=seg["data_offset"],
                   shape=(n_rg, n_az, 2))
    layout = {"data_offset": seg["data_offset"], "data_length": seg["data_length"],
              "dtype": dtype, "IMODE": sub["IMODE"], "IC": sub["IC"],
              "NBPR": sub["NBPR"], "NBPC": sub["NBPC"], "IID2": sub["IID2"].strip(),
              "ISORCE": sub["ISORCE"].strip(), "IDATIM": sub["IDATIM"]}
    return mm, layout


# --------------------------------------------------------------------------
# SICD XML
# --------------------------------------------------------------------------
class SicdXml:
    """
    Namespace-agnostic accessor over a SICD XML document.

    Parameters
    ----------
    xml_text : str
        The SICD XML document.
    """

    def __init__(self, xml_text):
        self.root = ET.fromstring(xml_text)
        tag = self.root.tag
        self.ns = tag.split("}")[0].strip("{") if "}" in tag else ""

    def _q(self, path):
        if not self.ns:
            return path
        return "/".join(f"{{{self.ns}}}{p}" for p in path.split("/"))

    def find(self, path):
        return self.root.find(self._q(path))

    def findall(self, path):
        return self.root.findall(self._q(path))

    def text(self, path, default=None):
        e = self.find(path)
        return default if e is None or e.text is None else e.text.strip()

    def num(self, path, default=None):
        t = self.text(path)
        return default if t is None else float(t)

    def int_(self, path, default=None):
        t = self.text(path)
        return default if t is None else int(t)

    def poly1d(self, path):
        """1-D polynomial as a coefficient array (index = exponent), or None."""
        e = self.find(path)
        if e is None:
            return None
        coef = np.zeros(int(e.get("order1")) + 1)
        for c in e:
            coef[int(c.get("exponent1"))] = float(c.text)
        return coef

    def poly2d(self, path):
        """2-D polynomial as an array ``coef[e1, e2]``, or None."""
        e = self.find(path)
        if e is None:
            return None
        coef = np.zeros((int(e.get("order1")) + 1, int(e.get("order2")) + 1))
        for c in e:
            coef[int(c.get("exponent1")), int(c.get("exponent2"))] = float(c.text)
        return coef

    def xyz_poly(self, path):
        """XYZ polynomial as an array of shape (3, order + 1)."""
        parts = [self.poly1d(f"{path}/{axis}") for axis in ("X", "Y", "Z")]
        out = np.zeros((3, max(len(p) for p in parts)))
        for i, p in enumerate(parts):
            out[i, :len(p)] = p
        return out


# --------------------------------------------------------------------------
# SICD -> radar grid, orbit, attitude, calibration LUTs
# --------------------------------------------------------------------------
def datetime_from_iso(text):
    """isce3 DateTime from an ISO 8601 string that may carry nanoseconds."""
    t = text.rstrip("Z")
    if "." in t:
        head, frac = t.split(".")
        frac = (frac + "000000000")[:9]
        return isce3.core.DateTime(head) + isce3.core.TimeDelta(int(frac) * 1e-9)
    return isce3.core.DateTime(t)


def _polyval_xyz(coef, t):
    """Evaluate a (3, order + 1) XYZ polynomial at times t -> (len(t), 3)."""
    return np.stack([np.polyval(c[::-1], t) for c in coef], axis=1)


def _polyder_xyz(coef):
    out = np.zeros((3, max(coef.shape[1] - 1, 1)))
    for i in range(3):
        d = np.polyder(coef[i][::-1])[::-1]
        out[i, :len(d)] = d
    return out


def build_grid(sicd):
    """
    Derive the radar grid and the scalar metadata from the SICD header.

    Parameters
    ----------
    sicd : SicdXml
        Parsed header.

    Returns
    -------
    dict
        Grid description; ``zero_doppler_time`` is in seconds since ``epoch``.

    Raises
    ------
    UnsupportedSicdError
        For a grid other than RGZERO / INCA / SLANT, a non-linear
        ``TimeCAPoly`` or a sub-image.
    """
    grid_type = sicd.text("Grid/Type")
    image_type = sicd.text("RMA/ImageType")
    image_plane = sicd.text("Grid/ImagePlane")
    algo = sicd.text("ImageFormation/ImageFormAlgo")
    if grid_type != "RGZERO" or image_type != "INCA" or image_plane != "SLANT":
        raise UnsupportedSicdError(
            f"Grid/Type={grid_type} RMA/ImageType={image_type} ImagePlane={image_plane} "
            f"ImageFormAlgo={algo}: only RGZERO / RMA-INCA / SLANT is supported; a "
            "polar-format (RGAZIM) grid needs new geometry, not a repackaging")

    time_ca = sicd.poly1d("RMA/INCA/TimeCAPoly")
    if time_ca is None or len(time_ca) != 2:
        order = None if time_ca is None else len(time_ca) - 1
        raise UnsupportedSicdError(
            f"RMA/INCA/TimeCAPoly order {order}: a uniform zero-Doppler time axis "
            "(order 1) is required")

    n_rg = sicd.int_("ImageData/NumRows")
    n_az = sicd.int_("ImageData/NumCols")
    if sicd.int_("ImageData/FirstRow") or sicd.int_("ImageData/FirstCol"):
        raise UnsupportedSicdError("ImageData/FirstRow or FirstCol != 0 (sub-image)")
    scp_row = sicd.int_("ImageData/SCPPixel/Row")
    scp_col = sicd.int_("ImageData/SCPPixel/Col")
    ss_rg = sicd.num("Grid/Row/SS")
    ss_az = sicd.num("Grid/Col/SS")
    r_ca_scp = sicd.num("RMA/INCA/R_CA_SCP")

    collect_start = datetime_from_iso(sicd.text("Timeline/CollectStart"))
    epoch = isce3.core.DateTime(collect_start.year, collect_start.month,
                                collect_start.day)
    cs_offset = (collect_start - epoch).total_seconds()

    # Closest-approach time after CollectStart, per SICD column.
    slope = float(time_ca[1]) * ss_az               # seconds per column
    t_col0 = float(time_ca[0]) - scp_col * slope   # time of column 0
    flip = slope < 0
    dt = abs(slope)
    # NISAR line l <-> SICD column c; with the flip c = n_az - 1 - l.
    t_line0 = t_col0 + (n_az - 1) * slope if flip else t_col0
    zero_doppler_time = cs_offset + t_line0 + dt * np.arange(n_az)
    slant_range = r_ca_scp + (np.arange(n_rg) - scp_row) * ss_rg

    side = sicd.text("SCPCOA/SideOfTrack")[0].upper()
    ucol = np.array([sicd.num(f"Grid/Col/UVectECF/{a}") for a in "XYZ"])

    fp_min = sicd.num("ImageFormation/TxFrequencyProc/MinProc")
    fp_max = sicd.num("ImageFormation/TxFrequencyProc/MaxProc")
    f_min = sicd.num("RadarCollection/TxFrequency/Min")
    f_max = sicd.num("RadarCollection/TxFrequency/Max")
    fc_proc = 0.5 * (fp_min + fp_max)

    ipp = []
    for s in sicd.findall("Timeline/IPP/Set"):
        e = s.find(sicd._q("IPPPoly"))
        coef = {int(c.get("exponent1")): float(c.text) for c in e}
        ipp.append(coef.get(1, np.nan))
    prf_acq = float(np.nanmean(ipp)) if ipp else 1.0 / dt

    return dict(
        n_rg=n_rg, n_az=n_az, scp_row=scp_row, scp_col=scp_col,
        ss_rg=ss_rg, ss_az=ss_az, r_ca_scp=r_ca_scp,
        collect_start=collect_start, epoch=epoch, cs_offset=cs_offset,
        collect_duration=sicd.num("Timeline/CollectDuration"),
        time_ca=time_ca, slope=slope, dt=dt, flip=flip,
        t_img=(t_line0, t_line0 + dt * (n_az - 1)),  # seconds after CollectStart
        zero_doppler_time=zero_doppler_time, slant_range=slant_range,
        lookside={"L": "Left", "R": "Right"}[side], ucol=ucol,
        fc_proc=fc_proc, fc_acq=0.5 * (f_min + f_max),
        bw_proc=fp_max - fp_min, bw_acq=f_max - f_min,
        wavelength=SPEED_OF_LIGHT / fc_proc,
        prf_acq=prf_acq, prf_grid=1.0 / dt,
        incidence_deg=sicd.num("SCPCOA/IncidenceAng"),
        az_bw=sicd.num("Grid/Col/ImpRespBW") / abs(float(time_ca[1])),
        chirp_duration=sicd.num("RadarCollection/Waveform/WFParameters/TxPulseLength"),
        chirp_slope=sicd.num("RadarCollection/Waveform/WFParameters/TxFMRate"),
        pol=sicd.text("ImageFormation/TxRcvPolarizationProc").replace(":", ""),
        scp_llh=(sicd.num("GeoData/SCP/LLH/Lon"), sicd.num("GeoData/SCP/LLH/Lat"),
                 sicd.num("GeoData/SCP/LLH/HAE")),
        scp_ecf=np.array([sicd.num(f"GeoData/SCP/ECF/{a}") for a in "XYZ"]),
        pixel_type=sicd.text("ImageData/PixelType"),
        grid_type=grid_type, image_type=image_type, algo=algo,
        processing_type=sicd.text("ImageFormation/Processing/Type"),
        processing_applied=sicd.text("ImageFormation/Processing/Applied"),
        noise_poly=sicd.poly2d("Radiometric/NoiseLevel/NoisePoly"),
        noise_type=sicd.text("Radiometric/NoiseLevel/NoiseLevelType"),
        beta_poly=sicd.poly2d("Radiometric/BetaZeroSFPoly"),
    )


def _poly2d(coef, x, y):
    """Evaluate a SICD 2-D polynomial, coef[i, j] * x**i * y**j, with broadcasting."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                               np.asarray(y, dtype=np.float64))
    return np.polynomial.polynomial.polyval2d(x, y, coef)


def image_xy(g, rows, cols):
    """SICD (row, col) indices -> image coordinates (xrow, ycol) in metres, rows x cols."""
    x = (np.asarray(rows, dtype=np.float64) - g["scp_row"]) * g["ss_rg"]
    y = (np.asarray(cols, dtype=np.float64) - g["scp_col"]) * g["ss_az"]
    return x[:, None], y[None, :]


def lut_xy(g, t_since_epoch, slant_range):
    """
    (zero-Doppler time since epoch, slant range) -> image coordinates (xrow, ycol).

    Independent of the azimuth flip: ``xrow = R - R_CA_SCP`` and
    ``TimeCAPoly(ycol) = t - CollectStart``.
    """
    x = np.asarray(slant_range, dtype=np.float64) - g["r_ca_scp"]
    t_rel = np.asarray(t_since_epoch, dtype=np.float64) - g["cs_offset"]
    y = (t_rel - float(g["time_ca"][0])) / float(g["time_ca"][1])
    return x, y


def clamp_to_image(g, x, y):
    """
    Clamp image coordinates to the image extent.

    SICD Radiometric polynomials describe the image only, while the calibration
    LUT grid extends beyond it; outside the image the LUTs take the value at
    the nearest image edge.
    """
    x_lo, x_hi = sorted((-g["scp_row"] * g["ss_rg"],
                         (g["n_rg"] - 1 - g["scp_row"]) * g["ss_rg"]))
    y_lo, y_hi = sorted((-g["scp_col"] * g["ss_az"],
                         (g["n_az"] - 1 - g["scp_col"]) * g["ss_az"]))
    return np.clip(x, x_lo, x_hi), np.clip(y, y_lo, y_hi)


def beta0_factor(g, x, y):
    """Radiometric/BetaZeroSFPoly(x, y), pixel power -> beta0; must be finite and > 0."""
    b = _poly2d(g["beta_poly"], x, y)
    if not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise UnsupportedSicdError(
            "Radiometric/BetaZeroSFPoly is non-finite or non-positive over the "
            f"evaluated domain (min {np.nanmin(b):.3e}); refusing to calibrate")
    return b


def noise_power(g, x, y):
    """Radiometric/NoiseLevel/NoisePoly (dB of pixel power) -> pixel power, or None."""
    if g["noise_poly"] is None or (g["noise_type"] or "").upper() != "ABSOLUTE":
        return None
    return 10.0 ** (_poly2d(g["noise_poly"], x, y) / 10.0)


def radiometric_luts(g, lut_az, lut_rg):
    """Noise LUT on the (lut_az x lut_rg) grid and its description, per g["radiometry"]."""
    tt, rr = np.meshgrid(lut_az, lut_rg, indexing="ij")
    lx, ly = clamp_to_image(g, *lut_xy(g, tt, rr))
    noise = noise_power(g, lx, ly)
    if noise is None:
        return np.zeros_like(tt), ("Noise equivalent backscatter: not available (SICD "
                                   "NoiseLevel absent or not ABSOLUTE); zeros")
    where = ("(xrow, ycol), evaluated inside the image and held at the nearest image "
             "edge outside it")
    if g["radiometry"] == "beta0":
        return noise * beta0_factor(g, lx, ly), (
            "Noise equivalent backscatter in beta0 power: 10^(Radiometric/NoiseLevel/"
            f"NoisePoly / 10) * Radiometric/BetaZeroSFPoly at {where}")
    return noise, ("Noise equivalent backscatter in DN^2 (pixel power): "
                   f"10^(Radiometric/NoiseLevel/NoisePoly / 10) at {where}")


def build_orbit(sicd, g, spacing, margin):
    """
    Sample ``Position/ARPPoly`` into an isce3 orbit.

    Parameters
    ----------
    sicd : SicdXml
    g : dict
        Output of `build_grid`; ``orbit_span`` is added to it.
    spacing : float
        State-vector spacing in seconds.
    margin : float
        Extension beyond the collect and the image span, in seconds.

    Returns
    -------
    isce3.core.Orbit
    """
    arp = sicd.xyz_poly("Position/ARPPoly")
    vel = _polyder_xyz(arp)
    t0 = min(g["t_img"][0], 0.0) - margin
    t1 = max(g["t_img"][1], g["collect_duration"]) + margin
    n = int(np.ceil((t1 - t0) / spacing)) + 1
    ts = t0 + spacing * np.arange(n)
    pos = _polyval_xyz(arp, ts)
    vel_v = _polyval_xyz(vel, ts)
    svs = [isce3.core.StateVector(
        datetime=g["collect_start"] + isce3.core.TimeDelta(float(t)),
        position=p.tolist(), velocity=v.tolist()) for t, p, v in zip(ts, pos, vel_v)]
    g["orbit_span"] = (float(ts[0]), float(ts[-1]))
    return isce3.core.Orbit(state_vectors=svs, date_time=g["epoch"], type="Custom")


def build_attitude(sicd, g, orbit, spacing):
    """
    Antenna-frame to ECEF quaternions from ``Antenna/Tx``, or a TCN fallback.

    Returns
    -------
    attitude : isce3.core.Attitude
    source : str
        Which header block the quaternions come from.
    """
    t0, t1 = g["orbit_span"]
    n = int(np.ceil((t1 - t0) / spacing)) + 1
    ts = t0 + spacing * np.arange(n)
    times = [g["cs_offset"] + float(t) for t in ts]
    quats = []
    if sicd.find("Antenna/Tx/XAxisPoly") is not None:
        source = "Antenna/Tx/{XAxisPoly,YAxisPoly}, Z = X x Y"
        ax = _polyval_xyz(sicd.xyz_poly("Antenna/Tx/XAxisPoly"), ts)
        ay = _polyval_xyz(sicd.xyz_poly("Antenna/Tx/YAxisPoly"), ts)
        for x, y in zip(ax, ay):
            x = x / np.linalg.norm(x)
            y = y - (x @ y) * x
            y = y / np.linalg.norm(y)
            quats.append(isce3.core.Quaternion(np.stack([x, y, np.cross(x, y)], axis=1)))
    else:
        source = "TCN frame from the orbit (Antenna block absent)"
        for t in times:
            p, v = (np.asarray(a) for a in orbit.interpolate(t))
            tt = v / np.linalg.norm(v)
            nn = -p / np.linalg.norm(p)
            cc = np.cross(nn, tt)
            cc = cc / np.linalg.norm(cc)
            nn = np.cross(tt, cc)
            quats.append(isce3.core.Quaternion(np.stack([tt, cc, nn], axis=1)))
    return isce3.core.Attitude(times, quats, g["epoch"]), source


def radar_band(fc):
    """IEEE radar band letter of a centre frequency in Hz."""
    for name, lo, hi in (("L", 1e9, 2e9), ("S", 2e9, 4e9), ("C", 4e9, 8e9),
                         ("X", 8e9, 12e9), ("Ku", 12e9, 18e9), ("Ka", 26.5e9, 40e9)):
        if lo <= fc < hi:
            return name
    return "Unknown"


def radar_grid(g):
    """isce3 RadarGridParameters of the output product."""
    return isce3.product.RadarGridParameters(
        float(g["zero_doppler_time"][0]), g["wavelength"], g["prf_grid"],
        float(g["slant_range"][0]), g["ss_rg"], g["lookside"], g["n_az"], g["n_rg"],
        g["epoch"])


# --------------------------------------------------------------------------
# RSLC writer
# --------------------------------------------------------------------------
def _str(group, name, value, desc=None):
    ds = group.create_dataset(name, data=np.bytes_(value))
    if desc:
        ds.attrs["description"] = np.bytes_(desc)
    return ds


def _num(group, name, value, units=None, desc=None, dtype=None):
    ds = group.create_dataset(name, data=np.asarray(value, dtype=dtype))
    if units:
        ds.attrs["units"] = np.bytes_(units)
    if desc:
        ds.attrs["description"] = np.bytes_(desc)
    return ds


def _vhat(orbit, t):
    _, v = orbit.interpolate(float(t))
    v = np.asarray(v)
    return v / np.linalg.norm(v)


def write_metadata(fid, sicd, g, orbit, att, att_source, in_path, xml_text, layout):
    """Write identification, orbit, attitude, swath metadata and the LUTs."""
    epoch_attr = f"seconds since {g['epoch'].isoformat()}"
    fid.attrs["Conventions"] = np.bytes_("CF-1.7")

    ident = fid.create_group(IDENT)
    _num(ident, "diagnosticModeFlag", 0, dtype=np.uint8)
    _str(ident, "isGeocoded", "False")
    ident.create_dataset("listOfFrequencies", data=np.bytes_(["A"]))
    _str(ident, "missionId", sicd.text("CollectionInfo/CollectorName"))
    _str(ident, "platformName", sicd.text("CollectionInfo/CollectorName"))
    _str(ident, "instrumentName", "SAR")
    _str(ident, "productType", "RSLC")
    _str(ident, "productVersion", "0.1.0")
    _str(ident, "productLevel", "L1")
    _str(ident, "productSpecificationVersion", "0.9.0")
    _str(ident, "processingType", "repackaging")
    _str(ident, "processingCenter",
         f"{sicd.text('ImageCreation/Site', '')} / "
         f"{sicd.text('ImageCreation/Application', '')}"
         " (SICD repackaged by isce3 share/nisar/examples/sicd_to_nisar_rslc.py)")
    _str(ident, "processingDateTime",
         datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat())
    _str(ident, "granuleId", sicd.text("CollectionInfo/CoreName", "None"))
    _str(ident, "radarBand", radar_band(g["fc_proc"]))
    _num(ident, "absoluteOrbitNumber", 0, dtype="u4")
    _num(ident, "trackNumber", 0, dtype=np.uint8)
    _num(ident, "frameNumber", 0, dtype=np.uint16)
    for name in ("isUrgentObservation", "isJointObservation", "isDithered",
                 "isMixedMode"):
        _str(ident, name, "False")
    _num(ident, "hasInputDataException", 0, dtype=np.uint8,
         desc="0: no input data anomalies")
    ident.create_dataset("plannedObservationId", data=np.bytes_(["0"]))
    ident.create_dataset("plannedDatatakeId", data=np.bytes_(["0"]))
    _str(ident, "lookDirection", g["lookside"])
    # Ascending/descending from the sign of the ECEF z-velocity at scene centre.
    t_mid = 0.5 * (g["zero_doppler_time"][0] + g["zero_doppler_time"][-1])
    _, v_mid = orbit.interpolate(float(t_mid))
    _str(ident, "orbitPassDirection", "Ascending" if v_mid[2] > 0 else "Descending")
    t_start = g["epoch"] + isce3.core.TimeDelta(float(g["zero_doppler_time"][0]))
    t_end = g["epoch"] + isce3.core.TimeDelta(float(g["zero_doppler_time"][-1]))
    _str(ident, "zeroDopplerStartTime", t_start.isoformat())
    _str(ident, "zeroDopplerEndTime", t_end.isoformat())

    orb_grp = fid.create_group(f"{META}/orbit")
    orbit.save_to_h5(orb_grp)
    _num(orb_grp, "acceleration", np.zeros_like(np.asarray(orbit.velocity)),
         units="meters per second squared", desc="Not provided by SICD; zeros")
    att_grp = fid.create_group(f"{META}/attitude")
    att.save_to_h5(att_grp)
    _str(att_grp, "attitudeType", "Custom",
         desc=f"Antenna-frame to ECEF quaternions (w, x, y, z) derived from {att_source}")
    _num(att_grp, "angularVelocity", np.zeros((att.size, 3)),
         units="radians per second", desc="Not provided by SICD; zeros")

    sw = fid.create_group(SWATHS)
    _num(sw, "zeroDopplerTime", g["zero_doppler_time"], units=epoch_attr,
         desc="CF compliant dimension associated with azimuth time")
    _num(sw, "zeroDopplerTimeSpacing", g["dt"], units="seconds",
         desc="Time interval in the along-track direction for raster layers")
    fa = sw.create_group("frequencyA")
    _num(fa, "slantRange", g["slant_range"], units="meters",
         desc="CF compliant dimension associated with slant range")
    _num(fa, "slantRangeSpacing", g["ss_rg"], units="meters")
    _num(fa, "acquiredCenterFrequency", g["fc_acq"], units="Hz")
    _num(fa, "processedCenterFrequency", g["fc_proc"], units="Hz")
    _num(fa, "centerFrequency", g["fc_proc"], units="Hz")
    _num(fa, "acquiredRangeBandwidth", g["bw_acq"], units="Hz")
    _num(fa, "processedRangeBandwidth", g["bw_proc"], units="Hz")
    _num(fa, "rangeBandwidth", g["bw_proc"], units="Hz")
    _num(fa, "nominalAcquisitionPRF", g["prf_acq"], units="Hz",
         desc="Mean slope of Timeline/IPP/Set/IPPPoly; the image grid rate is "
              "1 / zeroDopplerTimeSpacing")
    _num(fa, "chirpDuration", g["chirp_duration"], units="seconds")
    _num(fa, "chirpSlope", g["chirp_slope"], units="Hz per second")
    _num(fa, "sceneCenterGroundRangeSpacing",
         g["ss_rg"] / np.sin(np.radians(g["incidence_deg"])), units="meters")
    _num(fa, "sceneCenterAlongTrackSpacing", g["ss_az"], units="meters")
    _num(fa, "processedAzimuthBandwidth", g["az_bw"], units="Hz")
    _num(fa, "numberOfSubSwaths", 1, dtype="i8")
    fa.create_dataset("listOfPolarizations", data=np.array([g["pol"]], dtype="S2"))
    fa["listOfPolarizations"].attrs["description"] = np.bytes_(
        "List of processed polarization layers with frequency A")

    # Zero native-Doppler LUT and the inputs record.
    rg_pad, az_pad = 20 * 1000.0, 20 * 0.25
    lut_rg = np.arange(g["slant_range"][0] - rg_pad,
                       g["slant_range"][-1] + rg_pad + 1000.0, 1000.0)
    lut_az = np.arange(np.floor(g["zero_doppler_time"][0] - az_pad),
                       np.ceil(g["zero_doppler_time"][-1] + az_pad) + 0.25, 0.25)
    params = fid.create_group(f"{META}/processingInformation/parameters")
    for grp in (params, params.create_group("frequencyA")):
        _num(grp, "slantRange", lut_rg, units="meters",
             desc="Slant range dimension corresponding to processing information records")
        _num(grp, "zeroDopplerTime", lut_az, units=epoch_attr,
             desc="Zero doppler time dimension corresponding to processing information records")
    _num(params["frequencyA"], "dopplerCentroid",
         np.zeros((lut_az.size, lut_rg.size)), units="Hz",
         desc="2D LUT of Doppler Centroid for Frequency A -- identically zero: "
              "an RGZERO/INCA grid is a zero-Doppler grid by definition "
              f"(SCPCOA/DopplerConeAng = {sicd.num('SCPCOA/DopplerConeAng'):.9f} deg)")
    inputs = fid.create_group(f"{META}/processingInformation/inputs")
    _str(inputs, "sicdFile", os.path.basename(in_path))
    _str(inputs, "sicdSha256Header",
         hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
         desc="SHA-256 of the SICD XML text as extracted from the NITF")
    _str(inputs, "sicdXml", xml_text, desc="Verbatim SICD XML data extension segment")
    _str(inputs, "sicdVersion", sicd.ns)
    _str(inputs, "sicdGridType", f"{g['grid_type']} / {g['image_type']}")
    _str(inputs, "sicdImageFormAlgo", g["algo"])
    _str(inputs, "sicdProcessingType",
         f"{g['processing_type']} (Applied={g['processing_applied']})")
    _str(inputs, "azimuthFlipped", str(bool(g["flip"])),
         desc="True when sign(RMA/INCA/TimeCAPoly[1]) < 0: NISAR line l = SICD "
              "column (NumCols-1-l)")
    _num(inputs, "timeCAPolySlope", float(g["time_ca"][1]), units="seconds per meter")
    _num(inputs, "uColDotVhat", float(g["ucol"] @ _vhat(orbit, t_mid)), units="1",
         desc="Grid/Col/UVectECF . v/|v| at scene centre; -1 for a left-looking INCA grid")
    _str(inputs, "converter", "isce3 share/nisar/examples/sicd_to_nisar_rslc.py")
    _str(inputs, "isce3Version", isce3.__version__)
    _str(inputs, "nitfImageLayout", json.dumps(layout))

    # Calibration LUTs:
    #   dn    : pixels are SICD DN; the noise LUT is in DN^2; no geometry LUTs.
    #   beta0 : |pixel|^2 is beta0; the noise LUT is in beta0 power and
    #           calibrationInformation/geometry/beta0 is identically 1.
    tt = np.zeros((lut_az.size, lut_rg.size))
    noise, noise_desc = radiometric_luts(g, lut_az, lut_rg)
    beta0 = g["radiometry"] == "beta0"
    cal_root = fid.create_group(f"{META}/calibrationInformation")
    cal = cal_root.create_group("frequencyA")
    for name, data, desc in (
            ("elevationAntennaPattern", np.zeros_like(tt),
             "Complex two-way elevation antenna pattern (not in SICD; zeros)"),
            ("noiseEquivalentBackscatter", noise, noise_desc)):
        grp = cal.create_group(name)
        _num(grp, "slantRange", lut_rg, units="meters")
        _num(grp, "zeroDopplerTime", lut_az, units=epoch_attr)
        _num(grp, g["pol"], data.astype(np.float32), units="1", desc=desc)
    if beta0:
        geo = cal_root.create_group("geometry")
        _num(geo, "slantRange", lut_rg, units="meters")
        _num(geo, "zeroDopplerTime", lut_az, units=epoch_attr)
        _num(geo, "beta0", np.ones_like(tt, dtype=np.float32), units="1",
             desc="Radiometric calibration LUT for beta0: identically 1, the image is "
                  "already beta0-calibrated (DN * sqrt(Radiometric/BetaZeroSFPoly))")
    _str(inputs, "radiometricCalibration",
         "beta0: pixel = DN * sqrt(Radiometric/BetaZeroSFPoly(xrow, ycol))" if beta0
         else "none: pixel = SICD DN")


def write_image(fid, mm, g, block_lines, compress):
    """
    Transpose (and flip) the SICD pixels into the RSLC image; write the valid samples.

    With ``g["radiometry"] == "beta0"`` each pixel is multiplied by the real
    factor ``sqrt(BetaZeroSFPoly(xrow, ycol))`` at its own SICD (row, col).
    """
    n_rg, n_az, flip = g["n_rg"], g["n_az"], g["flip"]
    cpx = np.dtype([("r", np.float32), ("i", np.float32)])
    kw = dict(chunks=(min(512, n_az), min(512, n_rg)))
    if compress:
        kw.update(compression="gzip", compression_opts=1, shuffle=True)
    img = fid.create_dataset(f"{SWATHS}/frequencyA/{g['pol']}", shape=(n_az, n_rg),
                             dtype=cpx, **kw)
    beta0 = g["radiometry"] == "beta0"
    img.attrs["description"] = np.bytes_(
        f"Focused RSLC image ({g['pol']}), beta0-calibrated: SICD DN * "
        "sqrt(Radiometric/BetaZeroSFPoly(xrow, ycol))" if beta0 else
        f"Focused RSLC image ({g['pol']}), DN as in the SICD (no radiometric scaling)")
    img.attrs["units"] = np.bytes_("1")
    rows_all = np.arange(n_rg)
    valid = np.zeros((n_az, 2), dtype="i8")
    for l0 in range(0, n_az, block_lines):
        l1 = min(l0 + block_lines, n_az)
        if flip:
            c0, c1 = n_az - l1, n_az - l0          # columns of lines l0..l1-1, reversed
            blk = mm[:, c0:c1, :][:, ::-1, :]
            cols = np.arange(c1 - 1, c0 - 1, -1)  # SICD column of each output line
        else:
            blk = mm[:, l0:l1, :]
            cols = np.arange(l0, l1)
        blk = np.asarray(blk, dtype=np.float32)   # (range, lines, 2)
        nz = ((blk[..., 0] != 0) | (blk[..., 1] != 0)).T
        if beta0:
            bx, by = image_xy(g, rows_all, cols)
            blk = blk * np.sqrt(beta0_factor(g, bx, by)).astype(np.float32)[..., None]
        blk = np.ascontiguousarray(blk.transpose(1, 0, 2))  # (lines, range, 2)
        any_nz = nz.any(axis=1)
        valid[l0:l1, 0] = np.where(any_nz, np.argmax(nz, axis=1), 0)
        valid[l0:l1, 1] = np.where(any_nz, n_rg - np.argmax(nz[:, ::-1], axis=1), 0)
        img.write_direct(blk.view(cpx).reshape(l1 - l0, n_rg), dest_sel=np.s_[l0:l1])
    ds = fid[f"{SWATHS}/frequencyA"].create_dataset("validSamplesSubSwath1",
                                                     data=valid, dtype="i8")
    ds.attrs["description"] = np.bytes_(
        "First and last+1 valid range sample per line, derived from the data "
        "(first/last non-zero I/Q); ImageData/ValidData is not used")


def write_bounding_polygon(fid, g, orbit):
    """Write the perimeter on a constant-height surface at the SCP HAE."""
    dem = isce3.geometry.DEMInterpolator(float(g["scp_llh"][2]))
    poly = isce3.geometry.get_geo_perimeter_wkt(radar_grid(g), orbit,
                                                isce3.core.LUT2d(), dem)
    ds = fid[IDENT].create_dataset("boundingPolygon", data=np.bytes_(poly))
    ds.attrs["epsg"] = 4326
    ds.attrs["ogr_geometry"] = np.bytes_("polygon")
    ds.attrs["description"] = np.bytes_(
        f"Perimeter on a constant-height surface at the SCP HAE ({g['scp_llh'][2]:.1f} m)")
    return poly


# --------------------------------------------------------------------------
def load_header(xml_text, radiometry="dn", orbit_spacing=0.1, orbit_margin=1.0):
    """
    Parse a SICD header into the grid, orbit and attitude of the RSLC.

    Parameters
    ----------
    xml_text : str
        SICD XML document.
    radiometry : {"dn", "beta0"}
    orbit_spacing, orbit_margin : float
        See `build_orbit`.

    Returns
    -------
    sicd : SicdXml
    g : dict
    orbit : isce3.core.Orbit
    attitude : isce3.core.Attitude
    attitude_source : str
    """
    sicd = SicdXml(xml_text)
    g = build_grid(sicd)
    g["radiometry"] = radiometry
    if radiometry == "beta0" and g["beta_poly"] is None:
        raise UnsupportedSicdError("--radiometry beta0 needs Radiometric/BetaZeroSFPoly")
    orbit = build_orbit(sicd, g, orbit_spacing, orbit_margin)
    att, att_source = build_attitude(sicd, g, orbit, orbit_spacing)
    return sicd, g, orbit, att, att_source


def convert(in_path, out_path, radiometry="dn", block_lines=512, orbit_spacing=0.1,
            orbit_margin=1.0, compress=True):
    """
    Convert one SICD NITF into a NISAR RSLC HDF5 file.

    Returns
    -------
    dict
        Short summary of the product.
    """
    xml_text = read_sicd_xml(in_path)
    sicd, g, orbit, att, att_source = load_header(xml_text, radiometry,
                                                  orbit_spacing, orbit_margin)
    mm, layout = open_sicd_pixels(in_path, g["n_rg"], g["n_az"], g["pixel_type"])
    with h5py.File(out_path, "w", fs_strategy="page", fs_page_size=4194304) as fid:
        write_metadata(fid, sicd, g, orbit, att, att_source, in_path, xml_text, layout)
        write_image(fid, mm, g, block_lines, compress)
        write_bounding_polygon(fid, g, orbit)
    return {"output": out_path, "grid": f"{g['grid_type']}/{g['image_type']}",
            "shape_az_rg": [g["n_az"], g["n_rg"]], "polarization": g["pol"],
            "look_side": g["lookside"], "azimuth_flipped": bool(g["flip"]),
            "radiometry": radiometry}


def cmd_line_parser():
    parser = argparse.ArgumentParser(
        description="Convert an RGZERO / RMA-INCA SICD (NITF 2.1) into a NISAR RSLC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("sicd", help="input SICD NITF file")
    parser.add_argument("out", nargs="?", help="output RSLC HDF5 file")
    parser.add_argument("--radiometry", choices=("dn", "beta0"), default="dn",
                        help="dn: keep the SICD digital numbers; beta0: scale the pixels "
                             "by sqrt(Radiometric/BetaZeroSFPoly) and write the beta0 "
                             "calibration LUT that GCOV expects")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse the header and print the grid, write nothing")
    parser.add_argument("--xml", help="with --dry-run: read the SICD XML from this file "
                                      "instead of the NITF")
    parser.add_argument("--block-lines", type=int, default=512,
                        help="output lines per processing block")
    parser.add_argument("--orbit-spacing", type=float, default=0.1,
                        help="state-vector spacing in seconds")
    parser.add_argument("--orbit-margin", type=float, default=1.0,
                        help="orbit extension beyond the collect and the image, seconds")
    parser.add_argument("--no-compress", action="store_true",
                        help="write the image without gzip compression")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output file")
    return parser


def main(argv=None):
    args = cmd_line_parser().parse_args(argv)
    if args.xml and not args.dry_run:
        raise SystemExit("--xml is only meaningful with --dry-run")
    if not args.dry_run and not args.out:
        raise SystemExit("an output path is required unless --dry-run is given")
    if args.out and os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} exists (use --overwrite)")
    try:
        if args.dry_run:
            xml_text = (open(args.xml, encoding="utf-8").read() if args.xml
                        else read_sicd_xml(args.sicd))
            _, g, _, _, att_source = load_header(xml_text, args.radiometry,
                                                 args.orbit_spacing, args.orbit_margin)
            print(json.dumps({
                "grid": f"{g['grid_type']}/{g['image_type']}",
                "shape_az_rg": [g["n_az"], g["n_rg"]], "polarization": g["pol"],
                "look_side": g["lookside"], "azimuth_flipped": bool(g["flip"]),
                "zero_doppler_start": (g["epoch"] + isce3.core.TimeDelta(
                    float(g["zero_doppler_time"][0]))).isoformat(),
                "prf_grid_hz": g["prf_grid"],
                "slant_range_m": [float(g["slant_range"][0]), float(g["slant_range"][-1])],
                "wavelength_m": g["wavelength"], "attitude_source": att_source}, indent=1))
            return 0
        t0 = time.time()
        summary = convert(args.sicd, args.out, args.radiometry, args.block_lines,
                          args.orbit_spacing, args.orbit_margin, not args.no_compress)
    except UnsupportedSicdError as err:
        print(f"REJECTED: {err}", file=sys.stderr)
        return 1
    summary["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
