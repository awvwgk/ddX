import pyddx

import numpy as np

tobohr = 1 / 0.52917721092
charges = np.array([
    -0.04192, -0.04192, -0.04198, -0.04192, -0.04192, -0.04198,
    0.04193, 0.04193,  0.04197,  0.04193,  0.04193,  0.04197
])
rvdw = tobohr * np.array([
    4.00253, 4.00253, 4.00253, 4.00253, 4.00253, 4.00253,
    2.99956, 2.99956, 2.99956, 2.99956, 2.99956, 2.99956
])
centres = tobohr * np.array([
    [ 0.00000,  2.29035,  1.32281],  # noqa: E201
    [ 0.00000,  2.29035, -1.32281],  # noqa: E201
    [ 0.00000,  0.00000, -2.64562],  # noqa: E201
    [ 0.00000, -2.29035, -1.32281],  # noqa: E201
    [ 0.00000, -2.29035,  1.32281],  # noqa: E201
    [ 0.00000,  0.00000,  2.64562],  # noqa: E201
    [ 0.00103,  4.05914,  2.34326],  # noqa: E201
    [ 0.00103,  4.05914, -2.34326],  # noqa: E201
    [ 0.00000,  0.00000, -4.68652],  # noqa: E201
    [-0.00103, -4.05914, -2.34326],  # noqa: E201
    [-0.00103, -4.05914,  2.34326],  # noqa: E201
    [ 0.00000,  0.00000,  4.68652],  # noqa: E201
])

epsilon = 78.3553
model = pyddx.Pcm(charges, centres, rvdw, epsilon)
nuclear_mep = model.solute_nuclear_contribution()
solvation = model.compute(nuclear_mep)

eref = -0.00017974013712832552
if abs(eref - solvation.energy) > 1e-8:
    raise SystemExit(f"Large deviation:   {eref - solvation.energy:15.9g}")
