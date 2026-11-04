use Cwd qw(abs_path);
use File::Basename qw(dirname);

# latexmk evaluates rc contents, so __FILE__ does not identify this file.
my $report_dir = dirname(abs_path($rc_files_read[-1]));

# Use the bundled font subset without changing the system TeX installation.
$ENV{'TEXMFHOME'} = "$report_dir/texmf";
$do_cd = 1;
$pdf_mode = 1;
$out_dir = $report_dir;
@default_files = ("$report_dir/LaporanInsight.tex");
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -halt-on-error -synctex=1 %O %S';
