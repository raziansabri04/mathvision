"""
MathVision - Modul 1: generator template lembar.

Dua jenis lembar, memakai penanda ArUco, kotak, dan sistem koordinat yang sama:

A. Lembar JAWABAN (dipakai di kelas, satu lembar = satu foto)
     lembar_<JENJANG>_s<seed>.pdf   lembar soal A4 (cetak skala 100%)
     kunci_<JENJANG>_s<seed>.pdf    kunci jawaban
     layout_<JENJANG>_s<seed>.json  koordinat kotak + soal + kunci

B. Lembar PENGUMPULAN DATA (dipakai untuk melatih CNN, satu halaman = satu foto)
     data_<KODE>.pdf                2 halaman, tiap kelas 20 kali per penulis
     data_<KODE>_h<n>.json          koordinat kotak + label kelas tiap kotak

Sistem koordinat JSON: milimeter, titik asal pojok kiri-atas halaman,
sumbu y ke bawah. Ini sama dengan sistem koordinat citra setelah koreksi
perspektif (dikalikan px_per_mm), sehingga segment.py tinggal mengalikan.

Pemakaian (dari akar repo):
    python -m mv.template jawaban --jenjang SD  --n 10 --seed 1 --out templates
    python -m mv.template jawaban --jenjang SMP --n 10 --seed 1 --out templates
    python -m mv.template data --penulis 40 --out templates/data
    python -m mv.template data --kode P001  --out templates/data

Dependensi: opencv-python-headless, numpy, reportlab
(diuji dengan opencv 5.0.0, numpy 2.5.4, reportlab 5.0.1)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

# ----------------------------------------------------------------------
# Konstanta geometri (mm). Ubah di sini, semua modul lain ikut lewat JSON.
# ----------------------------------------------------------------------
A4_W, A4_H = 210.0, 297.0
MARGIN = 10.0          # jarak penanda ArUco ke tepi kertas
MARKER = 15.0          # sisi penanda ArUco (proposal: 15x15 mm)
BOX = 15.0             # sisi kotak jawaban (proposal: 15x15 mm)
GAP = 2.0              # jarak antar kotak
RIGHT_EDGE = 180.0     # tepi kanan kotak jawaban (rata kanan)
Q_Y0 = 70.0            # y atas baris soal pertama
Q_PITCH = 19.5         # jarak antar baris soal
MAX_SOAL = 10          # satu lembar = satu foto, maksimum 10 soal
PX_PER_MM = 8          # resolusi citra baku setelah koreksi perspektif
INSET_MM = 1.5         # saran: potong bagian dalam kotak, buang garis tepi
BOX_GRAY = 0.55        # abu-abu garis kotak (0=hitam). Perlu diuji pada cetakan nyata.
KAMUS_NAMA = "DICT_4X4_50"

JENJANG = {
    "SD":  {"peran": ["digit", "digit", "digit"],
            "judul": "Lembar Jawaban SD"},
    "SMP": {"peran": ["tanda", "digit", "digit", "digit"],
            "judul": "Lembar Jawaban SMP"},
}
TAMPIL_OP = {"+": "+", "-": "-", "*": "\u00d7"}  # \u00d7 = tanda kali

# Lembar pengumpulan data: 11 baris x 10 kolom = 110 kotak per halaman
DATA_KOLOM = 10
DATA_BARIS = 11
DATA_X0 = (A4_W - (DATA_KOLOM * BOX + (DATA_KOLOM - 1) * GAP)) / 2   # 21.0 mm
DATA_Y0 = 58.0         # y atas baris pertama
DATA_PITCH_Y = 19.0    # jarak antar baris (celah 4 mm memuat label kecil)
KELAS_NEGATIF = 10     # kelas 0-9 = digit, 10 = tanda negatif (11 kelas, sesuai CNN)


# ----------------------------------------------------------------------
# Soal
# ----------------------------------------------------------------------
def _fmt_operand(n: int) -> str:
    return f"({n})" if n < 0 else str(n)


def format_ekspresi(a: int, op: str, b: int) -> str:
    """Contoh: '-3 + 8', '5 - (-3)', '(-4) x (-3)'."""
    if op == "*":
        return f"{_fmt_operand(a)} {TAMPIL_OP[op]} {_fmt_operand(b)}"
    return f"{a} {TAMPIL_OP[op]} {_fmt_operand(b)}"


def _hitung(a: int, op: str, b: int) -> int:
    return {"+": a + b, "-": a - b, "*": a * b}[op]


def _soal_sd(rng: random.Random, nomor: int, p_simpan_pinjam: float) -> dict:
    """Operand 10-99, hasil tidak negatif. Sebagian soal dipaksa butuh
    menyimpan/meminjam, karena pola S1 dan S3 hanya bisa muncul di situ."""
    op = rng.choice(["+", "-"])
    butuh = rng.random() < p_simpan_pinjam
    while True:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        if op == "-" and a <= b:
            continue
        ada = (a % 10 + b % 10 >= 10) if op == "+" else (a % 10 < b % 10)
        if ada == butuh:
            break
    return {"nomor": nomor, "a": a, "op": op, "b": b}


def _soal_smp(rng: random.Random, nomor: int) -> dict:
    """+ dan -: operand -20..20. x: operand -12..12. Nol dihindari."""
    op = rng.choice(["+", "-", "*"])
    batas = 12 if op == "*" else 20
    nilai = [x for x in range(-batas, batas + 1) if x != 0]
    return {"nomor": nomor, "a": rng.choice(nilai), "op": op, "b": rng.choice(nilai)}


def buat_soal(jenjang: str, n_soal: int = 10, seed: int = 0,
              p_simpan_pinjam: float = 0.7) -> list[dict]:
    """Soal dibangkitkan dari seed, jadi lembar yang sama bisa dibuat ulang."""
    if jenjang not in JENJANG:
        raise ValueError(f"jenjang harus salah satu dari {list(JENJANG)}")
    if not 1 <= n_soal <= MAX_SOAL:
        raise ValueError(f"n_soal harus 1..{MAX_SOAL}")
    rng = random.Random(seed)
    hasil = []
    for nomor in range(1, n_soal + 1):
        s = (_soal_sd(rng, nomor, p_simpan_pinjam) if jenjang == "SD"
             else _soal_smp(rng, nomor))
        s["ekspresi"] = format_ekspresi(s["a"], s["op"], s["b"])
        s["kunci"] = _hitung(s["a"], s["op"], s["b"])
        hasil.append(s)
    return hasil


# ----------------------------------------------------------------------
# Penanda ArUco
# ----------------------------------------------------------------------
def _kamus():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, KAMUS_NAMA))


def _pola_marker(marker_id: int) -> np.ndarray:
    """Matriks 6x6 (1 = hitam): 4x4 bit data + 1 sel bingkai di tepi."""
    img = cv2.aruco.generateImageMarker(_kamus(), marker_id, 6, borderBits=1)
    return (img == 0).astype(np.uint8)


def _posisi_marker() -> dict[int, tuple[float, float]]:
    """ID 0 kiri-atas, 1 kanan-atas, 2 kanan-bawah, 3 kiri-bawah."""
    kiri, kanan = MARGIN, A4_W - MARGIN - MARKER
    atas, bawah = MARGIN, A4_H - MARGIN - MARKER
    return {0: (kiri, atas), 1: (kanan, atas), 2: (kanan, bawah), 3: (kiri, bawah)}


# ----------------------------------------------------------------------
# Layout (murni perhitungan, tanpa menggambar)
# ----------------------------------------------------------------------
def _penanda_json() -> list[dict]:
    penanda = []
    for mid, (x, y) in _posisi_marker().items():
        # urutan sudut = urutan keluaran detectMarkers: KiAt, KaAt, KaBw, KiBw
        sudut = [[x, y], [x + MARKER, y], [x + MARKER, y + MARKER], [x, y + MARKER]]
        penanda.append({"id": mid, "sudut_mm": sudut})
    return penanda


def hitung_layout(jenjang: str, soal: list[dict], seed: int = 0) -> dict:
    peran = JENJANG[jenjang]["peran"]
    n_kotak = len(peran)
    n_digit = peran.count("digit")
    lebar_total = n_kotak * BOX + (n_kotak - 1) * GAP
    x0 = RIGHT_EDGE - lebar_total

    kode_x0 = 62.0
    kode_siswa = [{"peran": "digit", "x": kode_x0 + i * (BOX + GAP), "y": 27.0,
                   "w": BOX, "h": BOX} for i in range(4)]

    daftar = []
    for s in soal:
        i = s["nomor"] - 1
        y = Q_Y0 + i * Q_PITCH
        # pastikan jawaban muat di kotak yang tersedia
        k = s["kunci"]
        if len(str(abs(k))) > n_digit or (k < 0 and "tanda" not in peran):
            raise ValueError(f"kunci {k} tidak muat di kotak {jenjang}")
        kotak = [{"peran": p, "x": x0 + j * (BOX + GAP), "y": y, "w": BOX, "h": BOX}
                 for j, p in enumerate(peran)]
        daftar.append({**s, "kotak": kotak})

    return {
        "versi": 1,
        "jenis": "jawaban",
        "jenjang": jenjang,
        "seed": seed,
        "keterangan": "Koordinat dalam mm, asal kiri-atas, y ke bawah.",
        "halaman_mm": [A4_W, A4_H],
        "px_per_mm": PX_PER_MM,
        "inset_mm": INSET_MM,
        "aruco": {"kamus": KAMUS_NAMA, "ukuran_mm": MARKER, "penanda": _penanda_json()},
        "kode_siswa": {"kotak": kode_siswa},
        "aturan_baca": {
            "rata_kanan": True,
            "peran_kotak": peran,
            "catatan": "Kotak kosong di sisi kiri diabaikan. Pada SMP, kotak "
                       "'tanda' terisi berarti bilangan negatif.",
        },
        "soal": daftar,
    }


# ----------------------------------------------------------------------
# Menggambar PDF
# ----------------------------------------------------------------------
def _rect(c, x, y, w, h, **kw):
    """Gambar persegi panjang; x, y, w, h dalam mm, y dari atas halaman."""
    c.rect(x * mm, (A4_H - y - h) * mm, w * mm, h * mm, **kw)


def _teks(c, x, y, s, font="Helvetica", size=10, align="left", gray=0.0):
    """Teks dengan baseline di y mm (dari atas)."""
    c.setFont(font, size)
    c.setFillGray(gray)
    px, py = x * mm, (A4_H - y) * mm
    if align == "center":
        c.drawCentredString(px, py, s)
    elif align == "right":
        c.drawRightString(px, py, s)
    else:
        c.drawString(px, py, s)


def _gambar_marker(c, marker_id: int, x: float, y: float):
    """Penanda digambar vektor agar tajam pada resolusi cetak berapa pun."""
    pola = _pola_marker(marker_id)
    sel = MARKER / 6
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    c.setLineWidth(0.1)  # menutup garis celah antar persegi
    for r in range(6):
        k = 0
        while k < 6:
            if pola[r, k]:
                j = k
                while j < 6 and pola[r, j]:
                    j += 1
                _rect(c, x + k * sel, y + r * sel, (j - k) * sel, sel, stroke=1, fill=1)
                k = j
            else:
                k += 1


def _gambar_kotak(c, k: dict):
    c.setStrokeGray(BOX_GRAY)
    c.setLineWidth(0.3 * mm)
    _rect(c, k["x"], k["y"], k["w"], k["h"], stroke=1, fill=0)


def _petunjuk(jenjang: str) -> list[str]:
    baris = [
        "Tulis SATU karakter per kotak, rata kanan (mulai dari kotak paling kanan).",
        "Gunakan pena atau pensil gelap (2B atau lebih pekat). Jangan menyentuh garis kotak.",
    ]
    if jenjang == "SMP":
        baris.append("Jawaban negatif: tulis tanda minus (-) di kotak pertama. "
                     "Jawaban positif: kosongkan kotak pertama.")
    else:
        baris.append("Kosongkan kotak yang tidak dipakai.")
    baris.append("Tulis kode siswa 4 digit pada kotak di atas. Jangan menulis nama.")
    return baris


def gambar_lembar(layout: dict, path_pdf: str | Path) -> None:
    jenjang = layout["jenjang"]
    c = canvas.Canvas(str(path_pdf), pagesize=A4)
    c.setTitle(f"MathVision {jenjang} seed {layout['seed']}")

    for p in layout["aruco"]["penanda"]:
        x, y = p["sudut_mm"][0]
        _gambar_marker(c, p["id"], x, y)

    _teks(c, A4_W / 2, 17.5, JENJANG[jenjang]["judul"], "Helvetica-Bold", 15, "center")
    _teks(c, 30, 36.5, "Kode siswa:", "Helvetica-Bold", 11)
    for k in layout["kode_siswa"]["kotak"]:
        _gambar_kotak(c, k)

    _teks(c, 30, 50, "Petunjuk", "Helvetica-Bold", 8)
    for i, baris in enumerate(_petunjuk(jenjang)):
        assert stringWidth(baris, "Helvetica", 7.5) <= 150 * mm, f"petunjuk terlalu panjang: {baris}"
        _teks(c, 30, 54 + i * 3.8, baris, "Helvetica", 7.5)

    # label kecil di atas kotak baris pertama
    nama_posisi = {"tanda": "tanda"}
    digit_label = iter(["ratusan", "puluhan", "satuan"][-layout["aturan_baca"]["peran_kotak"].count("digit"):])
    for k in layout["soal"][0]["kotak"]:
        teks = nama_posisi["tanda"] if k["peran"] == "tanda" else next(digit_label)
        _teks(c, k["x"] + k["w"] / 2, Q_Y0 - 1.5, teks, "Helvetica", 6, "center", gray=0.45)

    for s in layout["soal"]:
        y = s["kotak"][0]["y"]
        _teks(c, 30, y + 10, f"{s['nomor']}.", "Helvetica-Bold", 13)
        teks = f"{s['ekspresi']} ="
        x_batas = s["kotak"][0]["x"] - 4
        assert 42 + stringWidth(teks, "Helvetica", 16) / mm <= x_batas, f"soal terlalu panjang: {teks}"
        _teks(c, 42, y + 10, teks, "Helvetica", 16)
        for k in s["kotak"]:
            _gambar_kotak(c, k)

    _teks(c, A4_W / 2, 291.5, f"MathVision | {jenjang} | seed {layout['seed']}",
          "Helvetica", 6, "center", gray=0.5)
    c.showPage()
    c.save()


def gambar_kunci(layout: dict, path_pdf: str | Path) -> None:
    c = canvas.Canvas(str(path_pdf), pagesize=A4)
    c.setTitle(f"Kunci {layout['jenjang']} seed {layout['seed']}")
    _teks(c, 20, 25, f"Kunci Jawaban - {JENJANG[layout['jenjang']]['judul']}", "Helvetica-Bold", 14)
    _teks(c, 20, 32, f"seed {layout['seed']}", "Helvetica", 9, gray=0.4)
    for i, s in enumerate(layout["soal"]):
        _teks(c, 20, 48 + i * 10, f"{s['nomor']}.   {s['ekspresi']} = {s['kunci']}", "Helvetica", 13)
    c.showPage()
    c.save()


# ----------------------------------------------------------------------
# Lembar pengumpulan data (untuk melatih CNN)
# ----------------------------------------------------------------------
def label_kelas(kelas: int) -> str:
    """Teks label di atas kotak: '0'..'9', atau 'minus' untuk kelas negatif."""
    return "minus" if kelas == KELAS_NEGATIF else str(kelas)


def urutan_kelas(kode_penulis: str, ulangan: int = 20, seed: int = 0) -> list[int]:
    """Urutan kelas untuk satu penulis. Disusun per blok 11 (tiap kelas sekali,
    diacak), diulang `ulangan` kali. Hasilnya tiap kelas muncul tepat `ulangan`
    kali, dan urutan acak agar tulisan tidak monoton (0000000, 1111111, ...).
    Acak bergantung pada (seed, kode), jadi lembar yang sama bisa dibuat ulang."""
    rng = random.Random(f"{seed}|{kode_penulis}")
    urutan: list[int] = []
    for _ in range(ulangan):
        blok = list(range(11))
        rng.shuffle(blok)
        urutan += blok
    return urutan


def hitung_layout_data(kode_penulis: str, ulangan: int = 20, seed: int = 0) -> list[dict]:
    """Satu dict layout per halaman. Dengan ulangan=20: 220 kotak = 2 halaman."""
    if ulangan < 1:
        raise ValueError("ulangan harus >= 1")
    urutan = urutan_kelas(kode_penulis, ulangan, seed)
    per_hal = DATA_KOLOM * DATA_BARIS
    n_hal = -(-len(urutan) // per_hal)  # pembulatan ke atas

    halaman = []
    for h in range(n_hal):
        kotak = []
        for i, kelas in enumerate(urutan[h * per_hal:(h + 1) * per_hal]):
            baris, kolom = divmod(i, DATA_KOLOM)
            kotak.append({
                "peran": "karakter", "kelas": kelas, "label": label_kelas(kelas),
                "baris": baris, "kolom": kolom,
                "x": DATA_X0 + kolom * (BOX + GAP), "y": DATA_Y0 + baris * DATA_PITCH_Y,
                "w": BOX, "h": BOX,
            })
        halaman.append({
            "versi": 1,
            "jenis": "data",
            "kode_penulis": kode_penulis,
            "halaman": h + 1,
            "jumlah_halaman": n_hal,
            "seed": seed,
            "keterangan": "Koordinat dalam mm, asal kiri-atas, y ke bawah. "
                          "kelas 0-9 = digit, 10 = tanda negatif.",
            "halaman_mm": [A4_W, A4_H],
            "px_per_mm": PX_PER_MM,
            "inset_mm": INSET_MM,
            "aruco": {"kamus": KAMUS_NAMA, "ukuran_mm": MARKER, "penanda": _penanda_json()},
            "kotak": kotak,
        })
    return halaman


def _gambar_halaman_data(c, lay: dict) -> None:
    for p in lay["aruco"]["penanda"]:
        x, y = p["sudut_mm"][0]
        _gambar_marker(c, p["id"], x, y)

    _teks(c, A4_W / 2, 17.5, "Lembar Pengumpulan Data Tulisan Tangan", "Helvetica-Bold", 14, "center")
    _teks(c, 30, 32, f"Kode penulis: {lay['kode_penulis']}", "Helvetica-Bold", 11)
    _teks(c, 180, 32, f"Halaman {lay['halaman']} dari {lay['jumlah_halaman']}",
          "Helvetica", 10, "right")

    petunjuk = [
        "Tulis karakter yang tertera di atas setiap kotak. Satu karakter per kotak, di tengah kotak.",
        "Tulis seperti biasa, jangan dirapikan berlebihan. Gunakan pena atau pensil gelap (2B atau lebih pekat).",
        "Kotak berlabel 'minus' diisi tanda negatif: satu garis pendek mendatar (-).",
        "Jika salah tulis, coret kotak itu dengan tanda X besar. Kotak tercoret dibuang saat pengecekan.",
    ]
    _teks(c, 30, 37, "Petunjuk", "Helvetica-Bold", 8)
    for i, baris in enumerate(petunjuk):
        assert stringWidth(baris, "Helvetica", 7.5) <= 150 * mm, f"petunjuk terlalu panjang: {baris}"
        _teks(c, 30, 40.5 + i * 3.5, baris, "Helvetica", 7.5)

    for k in lay["kotak"]:
        _teks(c, k["x"] + k["w"] / 2, k["y"] - 1.2, k["label"], "Helvetica-Bold", 8,
              "center", gray=0.25)
        _gambar_kotak(c, k)

    _teks(c, A4_W / 2, 291.5,
          f"MathVision | data | {lay['kode_penulis']} | hal {lay['halaman']}/{lay['jumlah_halaman']}",
          "Helvetica", 6, "center", gray=0.5)


def gambar_lembar_data(halaman: list[dict], path_pdf: str | Path) -> None:
    c = canvas.Canvas(str(path_pdf), pagesize=A4)
    c.setTitle(f"MathVision data {halaman[0]['kode_penulis']}")
    for lay in halaman:
        _gambar_halaman_data(c, lay)
        c.showPage()
    c.save()


def buat_lembar_data(kode_penulis: str, ulangan: int = 20, seed: int = 0,
                     out_dir: str | Path = "templates/data") -> dict:
    """Satu PDF (semua halaman) + satu JSON per halaman untuk satu penulis."""
    halaman = hitung_layout_data(kode_penulis, ulangan, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"data_{kode_penulis}.pdf"
    gambar_lembar_data(halaman, pdf)
    json_paths = []
    for lay in halaman:
        jp = out / f"data_{kode_penulis}_h{lay['halaman']}.json"
        jp.write_text(json.dumps(lay, indent=2, ensure_ascii=False), encoding="utf-8")
        json_paths.append(jp)
    return {"pdf": pdf, "layout": json_paths}


# ----------------------------------------------------------------------
# Antarmuka utama
# ----------------------------------------------------------------------
def buat_lembar(jenjang: str, n_soal: int = 10, seed: int = 0,
                out_dir: str | Path = "templates") -> dict[str, Path]:
    """Buat lembar PDF, kunci PDF, dan layout JSON. Mengembalikan path-nya."""
    jenjang = jenjang.upper()
    soal = buat_soal(jenjang, n_soal, seed)
    layout = hitung_layout(jenjang, soal, seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nama = f"{jenjang}_s{seed}"
    paths = {"lembar": out / f"lembar_{nama}.pdf",
             "kunci": out / f"kunci_{nama}.pdf",
             "layout": out / f"layout_{nama}.json"}
    gambar_lembar(layout, paths["lembar"])
    gambar_kunci(layout, paths["kunci"])
    paths["layout"].write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Generator template MathVision")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pj = sub.add_parser("jawaban", help="lembar jawaban + kunci + layout JSON")
    pj.add_argument("--jenjang", choices=list(JENJANG), required=True)
    pj.add_argument("--n", type=int, default=10, help=f"jumlah soal (1-{MAX_SOAL})")
    pj.add_argument("--seed", type=int, default=0)
    pj.add_argument("--out", default="templates")

    pd = sub.add_parser("data", help="lembar pengumpulan data per penulis")
    g = pd.add_mutually_exclusive_group(required=True)
    g.add_argument("--penulis", type=int, help="buat untuk P001..P<N>")
    g.add_argument("--kode", help="buat untuk satu kode penulis, mis. P001")
    pd.add_argument("--ulangan", type=int, default=20, help="jumlah tulisan per kelas")
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--out", default="templates/data")

    a = ap.parse_args()
    if a.cmd == "jawaban":
        for nama, p in buat_lembar(a.jenjang, a.n, a.seed, a.out).items():
            print(f"{nama:7s} -> {p}")
    else:
        kode_list = [a.kode] if a.kode else [f"P{i:03d}" for i in range(1, a.penulis + 1)]
        for kode in kode_list:
            hasil = buat_lembar_data(kode, a.ulangan, a.seed, a.out)
            print(f"{kode}: {hasil['pdf']} ({len(hasil['layout'])} halaman)")


if __name__ == "__main__":
    main()
