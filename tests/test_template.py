"""
Uji Modul 1 (mv/template.py).

Jalankan dari akar repo:
    python -m unittest tests.test_template -v      # atau: pytest tests/

Uji geometri memerlukan poppler (perintah pdftoppm) untuk merender PDF.
Jika tidak ada, uji tersebut dilewati. Uji ini mengecek bahwa koordinat di JSON
benar-benar cocok dengan isi PDF, bukan hanya konsisten di dalam kode.
"""
import collections
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mv import template as T

ADA_POPPLER = shutil.which("pdftoppm") is not None


def render(pdf: Path, tmp: Path) -> np.ndarray:
    """Render PDF ke citra abu-abu pada PX_PER_MM piksel per mm."""
    prefix = tmp / "hal"
    subprocess.run(["pdftoppm", "-r", str(T.PX_PER_MM * 25.4), "-gray", "-png",
                    "-singlefile", str(pdf), str(prefix)], check=True)
    img = cv2.imread(str(prefix) + ".png", cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img


class TestSoal(unittest.TestCase):
    def test_reproducible(self):
        self.assertEqual(T.buat_soal("SD", 10, seed=7), T.buat_soal("SD", 10, seed=7))
        self.assertNotEqual(T.buat_soal("SD", 10, seed=7), T.buat_soal("SD", 10, seed=8))

    def test_batas_sd(self):
        for seed in range(50):
            for s in T.buat_soal("SD", 10, seed):
                self.assertTrue(10 <= s["a"] <= 99 and 10 <= s["b"] <= 99)
                self.assertTrue(0 <= s["kunci"] <= 999)
                self.assertIn(s["op"], "+-")

    def test_batas_smp(self):
        for seed in range(50):
            for s in T.buat_soal("SMP", 10, seed):
                batas = 12 if s["op"] == "*" else 20
                self.assertTrue(abs(s["a"]) <= batas and abs(s["b"]) <= batas)
                self.assertTrue(abs(s["kunci"]) <= 999)

    def test_kunci_benar(self):
        for j in ("SD", "SMP"):
            for s in T.buat_soal(j, 10, 3):
                self.assertEqual(s["kunci"], {"+": s["a"] + s["b"], "-": s["a"] - s["b"],
                                              "*": s["a"] * s["b"]}[s["op"]])

    def test_sd_memuat_simpan_pinjam(self):
        """Pola S1/S3 hanya bisa muncul bila ada soal yang butuh simpan/pinjam."""
        soal = T.buat_soal("SD", 10, seed=1)
        n = sum((s["a"] % 10 + s["b"] % 10 >= 10) if s["op"] == "+" else (s["a"] % 10 < s["b"] % 10)
                for s in soal)
        self.assertGreaterEqual(n, 3)

    def test_n_soal_tidak_valid(self):
        with self.assertRaises(ValueError):
            T.buat_soal("SD", 11)
        with self.assertRaises(ValueError):
            T.buat_soal("SMA", 5)


@unittest.skipUnless(ADA_POPPLER, "butuh pdftoppm (poppler)")
class TestGeometri(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _buat(self, jenjang):
        paths = T.buat_lembar(jenjang, 10, seed=1, out_dir=self.tmp)
        layout = json.loads(paths["layout"].read_text(encoding="utf-8"))
        return paths, layout, render(paths["lembar"], self.tmp)

    def test_empat_penanda_terdeteksi_di_posisi_benar(self):
        """KF-01: keempat penanda terdeteksi; posisi sudut cocok dengan JSON."""
        for j in ("SD", "SMP"):
            with self.subTest(jenjang=j):
                _, layout, img = self._buat(j)
                det = cv2.aruco.ArucoDetector(T._kamus(), cv2.aruco.DetectorParameters())
                corners, ids, _ = det.detectMarkers(img)
                self.assertIsNotNone(ids)
                self.assertEqual(sorted(ids.flatten().tolist()), [0, 1, 2, 3])
                for c, mid in zip(corners, ids.flatten()):
                    harapan = next(p for p in layout["aruco"]["penanda"] if p["id"] == mid)
                    harapan_px = np.array(harapan["sudut_mm"]) * T.PX_PER_MM
                    galat = np.abs(c[0] - harapan_px).max()
                    self.assertLess(galat, 2.0, f"penanda {mid}: galat {galat:.2f} px")

    def test_koordinat_kotak_cocok_dengan_pdf(self):
        """Garis tepi ada di koordinat JSON, bagian dalam kotak putih."""
        for j in ("SD", "SMP"):
            with self.subTest(jenjang=j):
                _, layout, img = self._buat(j)
                p = T.PX_PER_MM
                semua = [k for s in layout["soal"] for k in s["kotak"]] + layout["kode_siswa"]["kotak"]
                for k in semua:
                    x, y, w, h = (round(k[n] * p) for n in ("x", "y", "w", "h"))
                    for nama, tepi in (("atas", img[y - 3:y + 4, x + w // 2]),
                                       ("bawah", img[y + h - 3:y + h + 4, x + w // 2]),
                                       ("kiri", img[y + h // 2, x - 3:x + 4]),
                                       ("kanan", img[y + h // 2, x + w - 3:x + w + 4])):
                        self.assertLess(int(tepi.min()), 200, f"tepi {nama} tidak ditemukan di {k}")
                    ins = round(T.INSET_MM * p) + 3
                    dalam = img[y + ins:y + h - ins, x + ins:x + w - ins]
                    self.assertGreater(int(dalam.min()), 240, f"bagian dalam kotak tidak putih: {k}")

    def test_kotak_tidak_menabrak_penanda(self):
        """Ada jarak bebas >= 3 mm antara penanda dan kotak mana pun."""
        _, layout, _ = self._buat("SMP")
        semua = [k for s in layout["soal"] for k in s["kotak"]] + layout["kode_siswa"]["kotak"]
        for p in layout["aruco"]["penanda"]:
            (x0, y0), (x1, y1) = p["sudut_mm"][0], p["sudut_mm"][2]
            for k in semua:
                terpisah = (k["x"] > x1 + 3 or k["x"] + k["w"] < x0 - 3 or
                            k["y"] > y1 + 3 or k["y"] + k["h"] < y0 - 3)
                self.assertTrue(terpisah, f"kotak {k} terlalu dekat penanda {p['id']}")


class TestLembarData(unittest.TestCase):
    def test_jumlah_dan_keseimbangan_kelas(self):
        """Proposal: tiap kelas minimal 20 kali per penulis (~220 sampel)."""
        hal = T.hitung_layout_data("P001", ulangan=20, seed=0)
        self.assertEqual(len(hal), 2)
        semua = [k["kelas"] for h in hal for k in h["kotak"]]
        self.assertEqual(len(semua), 220)
        self.assertEqual(set(collections.Counter(semua).values()), {20})
        self.assertEqual(set(semua), set(range(11)))
        for h in hal:  # tiap halaman juga seimbang: 10 per kelas
            self.assertEqual(set(collections.Counter(k["kelas"] for k in h["kotak"]).values()), {10})

    def test_label_sesuai_kelas(self):
        for h in T.hitung_layout_data("P001"):
            for k in h["kotak"]:
                self.assertEqual(k["label"], "minus" if k["kelas"] == 10 else str(k["kelas"]))

    def test_reproducible_dan_beda_antar_penulis(self):
        a = T.hitung_layout_data("P001", seed=3)
        self.assertEqual(a, T.hitung_layout_data("P001", seed=3))
        self.assertNotEqual(T.urutan_kelas("P001"), T.urutan_kelas("P002"))

    def test_ulangan_bukan_kelipatan_10(self):
        hal = T.hitung_layout_data("P001", ulangan=25)
        self.assertEqual(sum(len(h["kotak"]) for h in hal), 275)
        self.assertEqual(len(hal), 3)

    def test_kotak_tidak_menabrak_penanda(self):
        for h in T.hitung_layout_data("P001"):
            for p in h["aruco"]["penanda"]:
                (x0, y0), (x1, y1) = p["sudut_mm"][0], p["sudut_mm"][2]
                for k in h["kotak"]:
                    terpisah = (k["x"] > x1 + 3 or k["x"] + k["w"] < x0 - 3 or
                                k["y"] > y1 + 3 or k["y"] + k["h"] < y0 - 3)
                    self.assertTrue(terpisah, f"kotak {k} terlalu dekat penanda {p['id']}")


@unittest.skipUnless(ADA_POPPLER, "butuh pdftoppm (poppler)")
class TestGeometriData(unittest.TestCase):
    def test_penanda_dan_kotak_cocok_dengan_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            hasil = T.buat_lembar_data("P001", out_dir=tmp)
            for n, jp in enumerate(hasil["layout"], start=1):
                with self.subTest(halaman=n):
                    lay = json.loads(jp.read_text(encoding="utf-8"))
                    prefix = tmp / f"h{n}"
                    subprocess.run(["pdftoppm", "-r", str(T.PX_PER_MM * 25.4), "-gray", "-png",
                                    "-singlefile", "-f", str(n), "-l", str(n),
                                    str(hasil["pdf"]), str(prefix)], check=True)
                    img = cv2.imread(str(prefix) + ".png", cv2.IMREAD_GRAYSCALE)

                    det = cv2.aruco.ArucoDetector(T._kamus(), cv2.aruco.DetectorParameters())
                    corners, ids, _ = det.detectMarkers(img)
                    self.assertIsNotNone(ids)
                    self.assertEqual(sorted(ids.flatten().tolist()), [0, 1, 2, 3])

                    p = T.PX_PER_MM
                    for k in lay["kotak"]:
                        x, y, w, h = (round(k[m] * p) for m in ("x", "y", "w", "h"))
                        for tepi in (img[y - 3:y + 4, x + w // 2], img[y + h - 3:y + h + 4, x + w // 2],
                                     img[y + h // 2, x - 3:x + 4], img[y + h // 2, x + w - 3:x + w + 4]):
                            self.assertLess(int(tepi.min()), 200, f"tepi tidak ditemukan di {k}")
                        ins = round(T.INSET_MM * p) + 3
                        dalam = img[y + ins:y + h - ins, x + ins:x + w - ins]
                        self.assertGreater(int(dalam.min()), 240, f"bagian dalam kotak tidak putih: {k}")


if __name__ == "__main__":
    unittest.main()
