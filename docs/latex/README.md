# Laporan LaTeX

Seluruh sumber, gambar, dan hasil kompilasi laporan berada di folder ini.

Jalankan dari root repositori:

```bash
make -C docs/latex
```

Alternatif tanpa `make`:

```bash
latexmk -r docs/latex/.latexmkrc
```

Diperlukan `latexmk` dan instalasi LaTeX dengan `pdflatex`, `fontenc`,
`inputenc`, `times`, `amsmath`, `graphicx`, `booktabs`, dan `url`.
Konfigurasi `.latexmkrc` menggunakan kelas IEEE dan subset font lokal,
sehingga kompilasi tidak memerlukan unduhan atau pemasangan paket tambahan
pada instalasi TeX Live yang digunakan proyek ini.

Hasil utama adalah `LaporanInsight.pdf`. Berkas bantu seperti `.aux`, `.log`,
`.fls`, `.fdb_latexmk`, dan `.synctex.gz` juga dibuat di folder ini dan
diabaikan Git. `make -C docs/latex clean` membersihkan berkas bantu tanpa
menghapus PDF.

```text
docs/latex/
├── LaporanInsight.tex
├── LaporanInsight.pdf          # hasil kompilasi, diabaikan Git
├── IEEEtran.cls
├── .latexmkrc
├── Makefile
├── figures/                    # empat gambar PDF sumber
├── texmf/                      # subset font untuk laporan
└── licenses/                   # lisensi dan asal dependensi lokal
```

## Dependensi lokal

- `IEEEtran.cls` versi 1.8b disalin tanpa perubahan dari
  [CTAN IEEEtran](https://ctan.org/pkg/ieeetran), dengan lisensi LPPL 1.3.
  Karya lengkap tersedia melalui
  [arsip IEEEtran](https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip).
  Pemberitahuan hak cipta dan kontribusi tetap tercantum dalam berkas kelas.
- `texmf/fonts/` berisi 26 berkas metrik, virtual font, dan Type 1 yang
  digunakan laporan, disalin tanpa perubahan dari paket Ubuntu
  [`texlive-fonts-recommended` versi `2021.20220204-1`](https://archive.ubuntu.com/ubuntu/pool/universe/t/texlive-base/texlive-fonts-recommended_2021.20220204-1_all.deb).
  SHA-256 paket: `76c7e04501ceb777a6257745f1b8f5261d06e10e5ea9ef9b24856b582fa35fc7`.
  Font Times/Courier didistribusikan dengan GPL; font URW menyertakan
  pengecualian untuk penyematan dalam dokumen PDF/PostScript.
  Lihat pemberitahuan asli dan salinan lisensi di `licenses/`.

Subset ini disediakan untuk membangun laporan yang ada. Jika laporan memakai
keluarga font lain, pasang paket TeX yang sesuai atau tambahkan dependensinya.
