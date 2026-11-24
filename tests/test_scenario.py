"""Instanzen, Handbeispiele und Kopien: gleiche Eingabe gleiche Instanz (auch in einem zweiten Prozess), Normierung, Träger, Randsummen, Kopien gegen ihre Quellen
(DEMO-PLAYBOOK §4: jede Kopie eines Kerns braucht einen Kopier-Test)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mab_constants as C
import mab_scenario as SC
from mab_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent


def test_gleiche_eingabe_gleiche_instanz_und_andere_eingabe_andere_instanz():
    z1 = SC.zonen(8, SC.instanz_rng(35, 1, 0))
    z2 = SC.zonen(8, SC.instanz_rng(35, 1, 0))
    z3 = SC.zonen(8, SC.instanz_rng(35, 1, 1))
    assert np.array_equal(z1.kosten, z2.kosten) and np.array_equal(z1.a, z2.a) and np.array_equal(z1.b, z2.b)
    assert not np.array_equal(z1.kosten, z3.kosten)


def test_zonen_sind_normiert_ganzzahlig_und_symmetrisch():
    for i in range(30):
        z = SC.zonen(10, SC.instanz_rng(1, 2, i))
        assert z.kosten.max() == 1.0 and np.allclose(z.kosten, z.kosten.T) and np.diag(z.kosten).max() == 0.0
        assert z.a.sum() == pytest.approx(1.0) and z.b.sum() == pytest.approx(1.0)
        a_int, b_int = z.a * z.gesamt, z.b * z.gesamt
        assert np.allclose(a_int, np.round(a_int)) and a_int.min() >= 10 and a_int.max() <= 99
        assert np.allclose(b_int, np.round(b_int)) and b_int.min() >= 1 and round(b_int.sum()) == z.gesamt


def test_traeger_hat_in_jeder_zeile_und_spalte_einen_eintrag_und_den_gewuenschten_nullanteil():
    reale = []
    for i in range(50):
        pat = SC.traeger(8, 0.5, SC.instanz_rng(1, 3, i))
        assert pat.any(axis=1).all() and pat.any(axis=0).all()
        reale.append(1 - pat.mean())
    assert 0.4 < np.mean(reale) < 0.5                                        # Reparatur füllt leere Zeilen und Spalten: etwas unter dem Nennwert


def test_basis_matrix_und_poisson_zaehlung():
    z = SC.zonen(10, SC.instanz_rng(1, 4, 0))
    T = SC.basis_matrix(z.kosten, SC.instanz_rng(1, 4, 1))
    assert (T > 0).all() and T.max() <= 1.5 and T.min() >= 0.5 * np.exp(-1 / 3)
    N = SC.poisson_matrix(z, SC.instanz_rng(1, 4, 2))
    assert (N >= 0).all() and np.allclose(N, np.round(N)) and (N.sum(1) > 0).all() and (N.sum(0) > 0).all()


def test_randsummen_arten():
    g = SC.instanz_rng(1, 5, 0)
    T0 = SC.basis_matrix(SC.zonen(6, g).kosten, g)
    for art in C.RANDSUMMEN:
        a, b = SC.randsummen(art, T0, SC.instanz_rng(1, 5, 1))
        assert a.sum() == pytest.approx(1.0) and b.sum() == pytest.approx(1.0) and (a > 0).all() and (b > 0).all()
    a, b = SC.randsummen(C.RS_GLEICH, T0, g)
    assert np.allclose(a, 1 / 6) and np.allclose(b, 1 / 6)
    a, b = SC.randsummen(C.RS_MILD, T0, SC.instanz_rng(1, 5, 1))
    assert (a / (T0.sum(1) / T0.sum())).min() > 0.8 and (a / (T0.sum(1) / T0.sum())).max() < 1.25     # Faktoren 0,9 bis 1,1 (nach Normierung etwas breiter)


def test_ansicht_hand_und_zonen():
    an = SC.ansicht(C.QUELLE_HAND, 6, 35, 0.0, 0.0)
    assert an.T0.tolist() == [[1, 2], [3, 4]] and an.a.tolist() == [5, 6] and an.b.tolist() == [5, 6] and an.kosten is None
    an = SC.ansicht(C.QUELLE_HAND, 6, 35, 0.0, 0.1)
    assert an.b.sum() == pytest.approx(1.1 * 11)
    an = SC.ansicht(C.QUELLE_ZONEN, 6, 35, 0.0, 0.0)
    assert an.T0.sum() == pytest.approx(C.GESAMT_FAHRTEN) and an.a.sum() == pytest.approx(an.b.sum()) and an.kosten.shape == (6, 6)
    an2 = SC.ansicht(C.QUELLE_ZONEN, 6, 35, 0.0, 0.05)
    assert an2.b.sum() == pytest.approx(1.05 * an2.a.sum()) and np.array_equal(an.T0, an2.T0)
    mit_nullen = SC.ansicht(C.QUELLE_ZONEN, 8, 35, 0.5, 0.0)
    assert (mit_nullen.T0 == 0).mean() > 0.3
    grav = SC.ansicht(C.QUELLE_ZONEN, 6, 35, 0.0, 0.0, gravitation=True)
    assert np.allclose(grav.a, np.round(grav.a)) and grav.a.sum() == pytest.approx(grav.b.sum())        # ganzzahlige Randsummen der Zonen


def test_zweiter_prozess_mit_anderem_pythonhashseed_rechnet_dieselbe_instanz():
    code = ("import sys; sys.path.insert(0, %r); import mab_scenario as SC, mab_scaling as S, numpy as np, mab_constants as C;"
            "an = SC.ansicht(C.QUELLE_ZONEN, 7, 35, 0.3, 0.0); r = S.abgleichen(an.T0, an.a, an.b);"
            "print(repr(float(an.T0.sum())), repr(float((an.T0 * np.arange(49).reshape(7, 7)).sum())), r.durchlaeufe, r.status)" % str(ROOT))
    ausgaben = set()
    for hashseed in ("1", "2"):
        res = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": hashseed, "PATH": "", "SYSTEMROOT": "C:\\Windows"}, timeout=120)
        assert res.returncode == 0, res.stderr
        ausgaben.add(res.stdout.strip())
    assert len(ausgaben) == 1


# ----------------------------------------------------------------------------------------------------------------------------------
# Kopien
# ----------------------------------------------------------------------------------------------------------------------------------

def test_rng_ist_eine_bytegleiche_kopie_der_vorlage_im_website_repo_falls_vorhanden():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden (Demo allein ausgecheckt)")
    assert (ROOT / "mab_rng.py").read_bytes() == vorlage.read_bytes()


def test_simplex_ist_eine_kopie_der_quelle_in_zuordnen_transportieren_demo_falls_vorhanden():
    quelle = ROOT.parent / "zuordnen-transportieren-demo"
    if not (quelle / "ztr_simplex.py").exists():
        pytest.skip("Quelle nicht vorhanden (Demo allein ausgecheckt)")
    original = (quelle / "ztr_simplex.py").read_text(encoding="utf-8").replace("ztr_", "mab_")
    assert (ROOT / "mab_simplex.py").read_text(encoding="utf-8") == original


def test_splitmix64_bekannte_ausgaben_fuer_seed_0():
    g = SplitMix64(0)
    assert [g.next_u64() for _ in range(4)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]


def test_poisson_mittelwert_und_varianz_passen_zu_lambda():
    g = SplitMix64(5)
    x = np.array([SC.poisson(g, 4.0) for _ in range(4000)])
    assert x.mean() == pytest.approx(4.0, abs=0.15) and x.var() == pytest.approx(4.0, abs=0.4)
