import numpy as np
import pytest

from collusion import games, stats


def test_true_mi_matches_hand_computation():
    # Values worked out by hand before the sprint (bits): CP 0.1, 0.2, 0.3, 1.0
    assert games.true_mi_assistant(0.1) == pytest.approx(0.0137, abs=5e-4)
    assert games.true_mi_assistant(0.2) == pytest.approx(0.0550, abs=5e-4)
    assert games.true_mi_assistant(0.3) == pytest.approx(0.1215, abs=5e-4)
    assert games.true_mi_assistant(1.0) == pytest.approx(np.log2(3))
    assert games.true_mi_assistant(0.0) == pytest.approx(0.0, abs=1e-12)


def test_cmi_deterministic_copy_is_log2_k():
    x = np.tile(np.arange(3), 100)
    s = np.zeros_like(x)
    assert stats._cmi_from_codes(x, x, s, 3, 3, 1) == pytest.approx(np.log2(3))
    # Copy holds within every stratum too.
    s2 = np.repeat(np.arange(4), 75)
    assert stats._cmi_from_codes(x, x, s2, 3, 3, 4) == pytest.approx(np.log2(3))


def test_cmi_conditional_independence_given_state():
    # X and Y both equal the state: dependent marginally, independent given S.
    rng = np.random.default_rng(1)
    s = rng.integers(0, 3, 3000)
    assert stats._cmi_from_codes(s, s, np.zeros_like(s), 3, 3, 1) == pytest.approx(np.log2(3), abs=0.01)
    assert stats._cmi_from_codes(s, s, s, 3, 3, 3) == pytest.approx(0.0, abs=1e-12)


def test_estimate_converges_to_analytic_mi():
    rng = np.random.default_rng(2)
    cfg = games.Config(cp=0.4, rounds=1000, games=20)
    a, gid = games.simulate(cfg, rng)
    st, ks = games.encode_states(a, gid, "none")
    sc = stats.pair_scores(a, st, 3, ks)
    assert sc["mi"][(0, 1)] == pytest.approx(games.true_mi_assistant(0.4), abs=0.01)
    assert sc["mi"][(2, 3)] < 0.002


def test_net_influence_three_player_identity():
    # With n = 3 the "max over others" is a single agent, so Gamma(3;2) = -Gamma(1;2) (0-indexed).
    rng = np.random.default_rng(3)
    a = rng.integers(0, 3, size=(500, 3))
    a[:, 1] = np.where(rng.random(500) < 0.5, a[:, 0], a[:, 1])
    g = stats.gamma_matrix(a, np.zeros(500, dtype=np.int64), 3, 1)
    G = stats.net_influence(g)
    assert G[2, 1] == pytest.approx(-G[0, 1])
    assert G[2, 0] == pytest.approx(-G[1, 0])


def test_global_confound_raises_cmi_but_not_net():
    rng = np.random.default_rng(4)
    a, gid = games.simulate(games.Config(common_cue=0.5, rounds=500, games=4), rng)
    st, ks = games.encode_states(a, gid, "none")
    sc = stats.pair_scores(a, st, 3, ks)
    assert min(sc["cmi"].values()) > 0.05
    assert max(sc["net"].values()) < 0.02


def test_history_reaction_confound_removed_by_state():
    rng = np.random.default_rng(5)
    a, gid = games.simulate(games.Config(react=0.6, rounds=2000, games=4), rng)
    s0, k0 = games.encode_states(a, gid, "none")
    s1, k1 = games.encode_states(a, gid, "profile")
    assert min(stats.pair_scores(a, s0, 3, k0)["cmi"].values()) > 0.05
    # residual is only plug-in bias: about ks*(k-1)^2 / (2 N ln 2) = 16*4/(2*8000*0.693) = 0.006
    assert max(stats.pair_scores(a, s1, 3, k1)["cmi"].values()) < 0.015


def test_permutation_detects_planted_pair_and_not_null():
    rng = np.random.default_rng(6)
    a, gid = games.simulate(games.Config(cp=0.6, rounds=30, games=20), rng)
    st, ks = games.encode_states(a, gid, "plurality")
    r = stats.permutation_test(a, st, 3, ks, n_perm=200, seed=0)["net"]
    assert r.top_pair == (0, 1) and r.top_p <= 0.01
    a0, gid0 = games.simulate(games.Config(rounds=30, games=20), np.random.default_rng(7))
    st0, _ = games.encode_states(a0, gid0, "plurality")
    r0 = stats.permutation_test(a0, st0, 3, ks, n_perm=200, seed=0)["net"]
    assert r0.top_p > 0.05


def test_profile_state_count():
    assert games.n_profile_states(4) == 1 + 15
    assert games.profile_state(np.array([0, 0, 1, 2]), 4) != games.profile_state(np.array([0, 1, 1, 2]), 4)


def test_leader_with_two_followers_separates_the_two_tests():
    # Player 0 is copied by players 1 and 2 alike, so min Gamma(0,1) = 0: dependence, but no pair
    # stands out (single seed; measured bootstrap FP rate here is 2.5–3.5% over 200 seeds). The
    # permutation test (H0: full independence) rejects; the bootstrap (H0: min Gamma
    # <= 0) does not. With a real planted pair the bootstrap does flag exactly that pair.
    a, gid = games.simulate(games.Config(leader=0.5, rounds=30, games=20), np.random.default_rng(10))
    st, ks = games.encode_states(a, gid, "profile")
    assert stats.permutation_test(a, st, 3, ks, n_perm=200, seed=0, scores=("net",))["net"].top_p <= 0.05
    assert stats.bootstrap_net_test(a, st, gid, 3, ks, n_boot=200, seed=0)["flagged"] == []
    a1, gid1 = games.simulate(games.Config(cp=0.5, rounds=30, games=20), np.random.default_rng(9))
    st1, _ = games.encode_states(a1, gid1, "profile")
    assert stats.bootstrap_net_test(a1, st1, gid1, 3, ks, n_boot=200, seed=0)["flagged"] == [(0, 1)]
