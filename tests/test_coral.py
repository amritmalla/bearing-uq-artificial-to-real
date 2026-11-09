import numpy as np

from bearing_uq.coral import align_held_out_bearings, coral_map


def test_coral_matches_source_moments():
    rng = np.random.default_rng(0)
    source = rng.multivariate_normal([0, 1, 2], [[1, .5, 0], [.5, 2, .3], [0, .3, .5]], size=4000)
    target = rng.multivariate_normal([3, -1, 0], [[4, -1, 0], [-1, 1, 0], [0, 0, 9]], size=4000)
    mapped = coral_map(target, source, target)
    np.testing.assert_allclose(mapped.mean(0), source.mean(0), atol=1e-9)
    np.testing.assert_allclose(np.cov(mapped, rowvar=False), np.cov(source, rowvar=False), rtol=0.01, atol=0.01)


def test_identical_domains_change_little():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 4))
    np.testing.assert_allclose(coral_map(x, x, x), x, atol=1e-2)


def test_held_out_bearing_does_not_use_its_own_windows():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(300, 2))
    test = np.vstack([rng.normal(size=(50, 2)), rng.normal(size=(50, 2)), rng.normal(5, 1, size=(50, 2))])
    bearing = np.repeat(["A", "B", "C"], 50)
    out = align_held_out_bearings(train, test, bearing)
    expected_a = coral_map(test[:50], train, test[50:])
    np.testing.assert_allclose(out[:50], expected_a)
    # Changing bearing A's own windows must not change how A is aligned (only its own rows move).
    test2 = test.copy()
    test2[:50] += 100
    out2 = align_held_out_bearings(train, test2, bearing)
    np.testing.assert_allclose(out2[:50], coral_map(test2[:50], train, test2[50:]))
    np.testing.assert_allclose(out2[:50] - out[:50], (test2[:50] - test[:50]) @ (
        coral_map(np.eye(2), train, test[50:]) - coral_map(np.zeros((2, 2)), train, test[50:])))
