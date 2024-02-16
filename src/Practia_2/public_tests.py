import numpy as np


def test_data():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    b_init = 785.1811367994083
    w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

    return X_train, y_train, w_init, b_init


def compute_cost_test(target):
    X_train, y_train, w_init, b_init = test_data()
    cost = target(X_train, y_train, w_init, b_init)
    target_cost = 1.5578904045996674e-12
    assert np.isclose(
        cost, target_cost, rtol=1e-4), f"Case 1: Cost must be {target_cost} for a perfect prediction but got {cost}"

    print("\033[92mAll tests passed!")


def compute_gradient_test(target):
    X_train, y_train, w_init, b_init = test_data()

    dj_db, dj_dw = target(X_train, y_train, w_init, b_init)
    #assert dj_dw.shape == w_init.shape, f"Wrong shape for dj_dw. {dj_dw} != {w_init.shape}"

    target_dj_db = -1.6739251122999121e-06
    target_dj_dw = [-2.73e-3, - 6.27e-6, - 2.22e-6, - 6.92e-5]

    assert np.isclose(dj_db, target_dj_db,
                      rtol=1e-4), f"Case 1: dj_db is wrong: {dj_db} != {target_dj_db}"
    assert np.allclose(
        dj_dw, target_dj_dw, rtol=1e-02), f"Case 1: dj_dw is wrong: {dj_dw} != {target_dj_dw}"

    print("\033[92mAll tests passed!")
