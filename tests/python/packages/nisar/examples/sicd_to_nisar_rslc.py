"""
Tests for share/nisar/examples/sicd_to_nisar_rslc.py.

The SICD headers are synthetic but geometrically consistent: a straight-line
ARP track at 7.5 km/s whose closest approach to the scene centre point (SCP)
is at a known time and range, for a left- and a right-looking collect. The
end-to-end cases write a small NITF 2.1 file (one 16-bit I/Q image segment and
one XML data extension segment) and read the product back with the isce3
RSLC reader.
"""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

import isce3
from nisar.products.readers import SLC

SCRIPT = (Path(__file__).resolve().parents[5] / "share" / "nisar" / "examples"
          / "sicd_to_nisar_rslc.py")
spec = importlib.util.spec_from_file_location("sicd_to_nisar_rslc_example", SCRIPT)
s2r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2r)

N_RG, N_AZ = 16, 32          # SICD NumRows (range), NumCols (azimuth)
SCP_ROW, SCP_COL = 7, 20
SS_RG, SS_AZ = 1.0, 1.25     # metres
SPEED = 7500.0               # m/s, straight-line ARP track
T_CA_SCP = 1.5               # s after CollectStart
COLLECT_START = "2026-01-02T03:04:05.123456789Z"
LAT, LON, HAE = np.radians(35.0), np.radians(139.0), 50.0
BETA_POLY = [[2.0e-4, 1.0e-7], [3.0e-7, 0.0]]
NOISE_POLY = [[-20.0, 0.01], [0.02, 0.0]]


def _geometry(side):
    """SCP, ARP at closest approach, velocity and Grid/Col unit vector."""
    ell = isce3.core.Ellipsoid()
    scp = np.asarray(ell.lon_lat_to_xyz(np.array([LON, LAT, HAE])))
    up = np.asarray(ell.n_vector(LON, LAT))
    north = np.array([-np.sin(LAT) * np.cos(LON), -np.sin(LAT) * np.sin(LON),
                      np.cos(LAT)])
    right = np.cross(north, up)      # east, for a track flying north
    look = 1.0 if side == "R" else -1.0
    arp_ca = scp - look * 400e3 * right + 500e3 * up
    vel = SPEED * north
    # INCA: Grid/Col/UVectECF = -look_sicd * v_hat, with look_sicd = +1 for left.
    ucol = look * north
    return scp, arp_ca, vel, ucol


def _poly1d(tag, coefs):
    body = "".join(f'<Coef exponent1="{i}">{c!r}</Coef>' for i, c in enumerate(coefs))
    return f'<{tag} order1="{len(coefs) - 1}">{body}</{tag}>'


def _poly2d(tag, coefs):
    coefs = np.asarray(coefs, dtype=float)
    body = "".join(f'<Coef exponent1="{i}" exponent2="{j}">{coefs[i, j]!r}</Coef>'
                   for i in range(coefs.shape[0]) for j in range(coefs.shape[1]))
    return (f'<{tag} order1="{coefs.shape[0] - 1}" order2="{coefs.shape[1] - 1}">'
            f'{body}</{tag}>')


def _xyz(tag, v):
    return f"<{tag}><X>{v[0]!r}</X><Y>{v[1]!r}</Y><Z>{v[2]!r}</Z></{tag}>"


def sicd_xml(side, grid_type="RGZERO", time_ca_order=1):
    """A consistent RGZERO / INCA SICD header for a left- or right-looking collect."""
    scp, arp_ca, vel, ucol = _geometry(side)
    arp0 = arp_ca - vel * T_CA_SCP
    arp_poly = "".join(_poly1d(axis, [arp0[i], vel[i]]) for i, axis in enumerate("XYZ"))
    slope = (1.0 if side == "R" else -1.0) / SPEED
    time_ca = [T_CA_SCP, slope] + ([1e-9] if time_ca_order == 2 else [])
    r_ca_scp = float(np.linalg.norm(scp - arp_ca))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<SICD xmlns="urn:SICD:1.3.0">
 <CollectionInfo><CollectorName>TEST1</CollectorName>
  <CoreName>TEST_CORE</CoreName></CollectionInfo>
 <ImageCreation><Application>unit test</Application><Site>none</Site></ImageCreation>
 <ImageData><PixelType>RE16I_IM16I</PixelType><NumRows>{N_RG}</NumRows><NumCols>{N_AZ}</NumCols>
  <FirstRow>0</FirstRow><FirstCol>0</FirstCol>
  <SCPPixel><Row>{SCP_ROW}</Row><Col>{SCP_COL}</Col></SCPPixel></ImageData>
 <GeoData><SCP>{_xyz("ECF", scp)}<LLH><Lat>{np.degrees(LAT)!r}</Lat><Lon>{np.degrees(LON)!r}</Lon>
  <HAE>{HAE!r}</HAE></LLH></SCP></GeoData>
 <Grid><ImagePlane>SLANT</ImagePlane><Type>{grid_type}</Type>
  <Row><SS>{SS_RG!r}</SS><ImpRespBW>1.2</ImpRespBW></Row>
  <Col>{_xyz("UVectECF", ucol)}<SS>{SS_AZ!r}</SS><ImpRespBW>0.7</ImpRespBW></Col></Grid>
 <Timeline><CollectStart>{COLLECT_START}</CollectStart><CollectDuration>3.0</CollectDuration>
  <IPP size="1"><Set index="1">{_poly1d("IPPPoly", [0.0, 9000.0])}</Set></IPP></Timeline>
 <Position><ARPPoly>{arp_poly}</ARPPoly></Position>
 <RadarCollection><TxFrequency><Min>9.55e9</Min><Max>9.75e9</Max></TxFrequency>
  <Waveform size="1"><WFParameters index="1"><TxPulseLength>1e-05</TxPulseLength>
  <TxFMRate>2e13</TxFMRate></WFParameters></Waveform></RadarCollection>
 <ImageFormation><TxRcvPolarizationProc>H:H</TxRcvPolarizationProc>
  <TxFrequencyProc><MinProc>9.55e9</MinProc><MaxProc>9.75e9</MaxProc></TxFrequencyProc>
  <ImageFormAlgo>RMA</ImageFormAlgo></ImageFormation>
 <SCPCOA><SideOfTrack>{side}</SideOfTrack><IncidenceAng>45.0</IncidenceAng>
  <DopplerConeAng>90.0</DopplerConeAng></SCPCOA>
 <Radiometric><NoiseLevel><NoiseLevelType>ABSOLUTE</NoiseLevelType>
  {_poly2d("NoisePoly", NOISE_POLY)}
  </NoiseLevel>{_poly2d("BetaZeroSFPoly", BETA_POLY)}</Radiometric>
 <RMA><ImageType>INCA</ImageType><INCA>{_poly1d("TimeCAPoly", time_ca)}
  <R_CA_SCP>{r_ca_scp!r}</R_CA_SCP></INCA></RMA>
</SICD>
"""


def _field(value, width):
    b = value.encode("ascii") if isinstance(value, str) else value
    assert len(b) <= width, (value, width)
    return b.ljust(width, b" ")


def write_nitf(path, xml, pixels):
    """
    Write a minimal NITF 2.1 file: one uncompressed 16-bit I/Q image segment
    and one XML data extension segment.

    ``pixels`` has shape (NumRows, NumCols, 2), SICD orientation.
    """
    n_rg, n_az, _ = pixels.shape
    data = np.ascontiguousarray(pixels.astype(">i2")).tobytes()
    values = {"FHDR": "NITF", "FVER": "02.10", "CLEVEL": "03", "STYPE": "BF01",
              "OSTAID": "TEST", "FDT": "20260101000000", "FSCLAS": "U",
              "FSCOP": "00000", "FSCPYS": "00000", "ENCRYP": "0", "FBKGC": "000"}
    im_values = {"IM": "IM", "IDATIM": "20260101000000", "ISCLAS": "U", "ENCRYP": "0",
                 "NROWS": f"{n_rg:08d}", "NCOLS": f"{n_az:08d}", "PVTYPE": "SI",
                 "IREP": "NODISPLY", "ICAT": "SAR", "ABPP": "16", "PJUST": "R",
                 "ICORDS": " "}
    imsub = b"".join(_field(im_values.get(name, ""), width)
                     for name, width in s2r._NITF_IMAGE_SUBHEADER)
    imsub += b"0" + b"NC" + b"2"                               # NICOM, IC, NBANDS
    for band in ("I", "Q"):
        imsub += _field("", 2) + _field(band, 6) + b"N" + _field("", 3) + b"0"
    imsub += (b"0" + b"P" + b"0001" + b"0001" + f"{n_az:04d}".encode()
              + f"{n_rg:04d}".encode() + b"16" + b"001" + b"000" + b"0000000000"
              + b"1.0 " + b"00000" + b"00000")
    dessub = _field("DE", 2) + _field("XML_DATA_CONTENT", 25) + _field("01", 2)
    dessub = _field(dessub + b"U", 200)
    xml_bytes = xml.encode("utf-8")

    def header(fl, hl):
        v = dict(values, FL=f"{fl:012d}", HL=f"{hl:06d}")
        h = b"".join(_field(v.get(name, ""), width) for name, width in s2r._NITF_FILE_HEADER)
        h += b"001" + f"{len(imsub):06d}".encode() + f"{len(data):010d}".encode()
        h += b"000" + b"000" + b"000"                            # NUMS, NUMX, NUMT
        h += b"001" + f"{len(dessub):04d}".encode() + f"{len(xml_bytes):09d}".encode()
        h += b"000" + b"00000" + b"00000"                        # NUMRES, UDHDL, XHDL
        return h

    hl = len(header(0, 0))
    fl = hl + len(imsub) + len(data) + len(dessub) + len(xml_bytes)
    with open(path, "wb") as fh:
        fh.write(header(fl, hl) + imsub + data + dessub + xml_bytes)


def _pattern_pixels():
    rows, cols = np.meshgrid(np.arange(N_RG), np.arange(N_AZ), indexing="ij")
    return np.stack([100 * rows + cols + 1, -(cols + 1)], axis=-1).astype(np.int16)


@pytest.mark.parametrize("side", ["L", "R"])
def test_grid(side):
    _, g, _, _, source = s2r.load_header(sicd_xml(side))
    assert g["flip"] == (side == "L")
    assert g["lookside"] == {"L": "Left", "R": "Right"}[side]
    assert g["n_az"] == N_AZ and g["n_rg"] == N_RG
    dt = SS_AZ / SPEED
    assert g["dt"] == pytest.approx(dt, rel=1e-15)
    assert g["prf_grid"] == pytest.approx(1.0 / dt, rel=1e-12)
    assert np.all(np.diff(g["zero_doppler_time"]) > 0)
    scp_line = N_AZ - 1 - SCP_COL if side == "L" else SCP_COL
    t_scp = g["cs_offset"] + T_CA_SCP
    assert g["zero_doppler_time"][scp_line] == pytest.approx(t_scp, abs=1e-9)
    # Nanoseconds of CollectStart survive (epoch = midnight UTC).
    assert g["cs_offset"] == pytest.approx(3 * 3600 + 4 * 60 + 5.123456789, abs=1e-9)
    assert g["slant_range"][SCP_ROW] == pytest.approx(g["r_ca_scp"], abs=1e-9)
    assert g["slant_range"][0] == pytest.approx(g["r_ca_scp"] - SCP_ROW * SS_RG,
                                                abs=1e-9)
    assert g["pol"] == "HH"
    assert source.startswith("TCN")


@pytest.mark.parametrize("side", ["L", "R"])
def test_scp_round_trip(side):
    """The SCP pixel projects onto GeoData/SCP/ECF and back through isce3."""
    _, g, orbit, _, _ = s2r.load_header(sicd_xml(side))
    grid = s2r.radar_grid(g)
    scp_line = N_AZ - 1 - SCP_COL if side == "L" else SCP_COL
    t = grid.sensing_start + scp_line / grid.prf
    r = grid.starting_range + SCP_ROW * grid.range_pixel_spacing
    xyz = isce3.geometry.rdr2geo_bracket(t, r, orbit, grid.lookside, 0.0,
                                         grid.wavelength,
                                         isce3.geometry.DEMInterpolator(HAE))
    assert np.linalg.norm(np.asarray(xyz) - g["scp_ecf"]) < 1e-3
    t2, r2 = isce3.geometry.geo2rdr_bracket(g["scp_ecf"], orbit, isce3.core.LUT2d(),
                                            grid.wavelength, grid.lookside)
    assert (t2 - grid.sensing_start) * grid.prf == pytest.approx(scp_line, abs=1e-4)
    assert (r2 - grid.starting_range) / grid.range_pixel_spacing == pytest.approx(
        SCP_ROW, abs=1e-4)


def test_rgazim_is_rejected():
    with pytest.raises(s2r.UnsupportedSicdError, match="RGAZIM"):
        s2r.load_header(sicd_xml("R", grid_type="RGAZIM"))


def test_nonlinear_time_ca_is_rejected():
    with pytest.raises(s2r.UnsupportedSicdError, match="order 2"):
        s2r.load_header(sicd_xml("R", time_ca_order=2))


def test_beta0_luts():
    _, g, _, _, _ = s2r.load_header(sicd_xml("R"), radiometry="beta0")
    t = g["zero_doppler_time"][[0, -1]]
    r = g["slant_range"][[0, -1]]
    noise, desc = s2r.radiometric_luts(g, t, r)
    x, y = s2r.clamp_to_image(g, *s2r.lut_xy(g, *np.meshgrid(t, r, indexing="ij")))
    expected = 10.0 ** (np.polynomial.polynomial.polyval2d(x, y, NOISE_POLY) / 10.0)
    expected *= np.polynomial.polynomial.polyval2d(x, y, BETA_POLY)
    np.testing.assert_allclose(noise, expected, rtol=1e-12)
    assert "beta0 power" in desc
    g["beta_poly"] = -np.asarray(BETA_POLY)
    with pytest.raises(s2r.UnsupportedSicdError, match="BetaZeroSFPoly"):
        s2r.beta0_factor(g, 0.0, 0.0)


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("radiometry", ["dn", "beta0"])
def test_end_to_end(tmp_path, side, radiometry):
    xml = sicd_xml(side)
    pixels = _pattern_pixels()
    pixels[:, 3, :] = 0                   # an all-zero SICD column (one output line)
    pixels[:2, :, :] = 0                  # first two range samples empty everywhere
    ntf = tmp_path / "scene.ntf"
    out = tmp_path / "rslc.h5"
    write_nitf(ntf, xml, pixels)
    summary = s2r.convert(str(ntf), str(out), radiometry=radiometry, block_lines=7)
    assert summary["azimuth_flipped"] == (side == "L")

    _, g, _, _, _ = s2r.load_header(xml, radiometry=radiometry)
    cols = np.arange(N_AZ)[::-1] if side == "L" else np.arange(N_AZ)
    expected = (pixels[..., 0] + 1j * pixels[..., 1]).astype(np.complex64)[:, cols].T
    if radiometry == "beta0":
        x, y = s2r.image_xy(g, np.arange(N_RG), cols)
        scale = np.sqrt(np.polynomial.polynomial.polyval2d(
            *np.broadcast_arrays(x, y), BETA_POLY)).astype(np.float32)
        expected = expected * scale.T
    with h5py.File(out, "r") as f:
        image = f[f"{s2r.SWATHS}/frequencyA/HH"][()]
        if image.dtype.names:             # h5py may return the (r, i) compound as is
            image = image["r"] + 1j * image["i"]
        np.testing.assert_allclose(image, expected, rtol=1e-6, atol=0)
        valid = f[f"{s2r.SWATHS}/frequencyA/validSamplesSubSwath1"][()]
        zero_line = N_AZ - 1 - 3 if side == "L" else 3
        assert tuple(valid[zero_line]) == (0, 0)
        others = np.delete(valid, zero_line, axis=0)
        assert np.all(others[:, 0] == 2) and np.all(others[:, 1] == N_RG)
        has_beta0 = f"{s2r.META}/calibrationInformation/geometry/beta0" in f
        assert has_beta0 == (radiometry == "beta0")
        assert f[f"{s2r.IDENT}/lookDirection"][()].decode() == g["lookside"]
        assert f[f"{s2r.IDENT}/radarBand"][()].decode() == "X"

    rslc = SLC(hdf5file=str(out))
    grid = rslc.getRadarGrid("A")
    ref = s2r.radar_grid(g)
    assert grid.length == N_AZ and grid.width == N_RG
    assert grid.sensing_start == pytest.approx(ref.sensing_start, abs=1e-9)
    assert grid.starting_range == pytest.approx(ref.starting_range, abs=1e-9)
    assert grid.prf == pytest.approx(ref.prf, rel=1e-12)
    assert grid.lookside == ref.lookside
    orbit = rslc.getOrbit()
    assert orbit.reference_epoch == g["epoch"]
