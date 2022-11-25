import numpy as np
from scipy import optimize, stats

from . import term_structure


class ChhCalibration:
    def __init__(
        self, price_grid: np.ndarray, discount_curve: term_structure.DiscountCurve
    ):
        self.price_grid = price_grid
        self.tenor = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
        self.expiry = np.array(
            [
                1,
                18 / 12,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                15,
                20,
                25,
                30,
            ]
        )
        self.df = discount_curve.df
        self._calibrate()

    def _calibrate(self) -> None:
        """
        chh calibration scheme

        """
        psi = np.zeros([len(self.df), len(self.df)])
        for i in range(len(psi)):
            psi[i] = self.df / self.df[i]
        self.v = np.zeros(psi.shape)
        self.chi = np.zeros(psi.shape)

        eq_21 = lambda wk, k: wk * np.sum(k) + k[len(k) - 1] - 1  # solve for swap rate wk
        eq_20 = lambda v, k: (
            k * stats.norm.cdf(v / 2) - k * stats.norm.cdf(-v / 2)
        )  # eq for chi, solve for v
        eq_19 = (
            lambda chi_j_k, chi, price, df, wk: df
            * (wk * (np.sum(chi) + chi_j_k) + chi_j_k)
            - price
        )  # eq for price, solve for chi

        # main scheme for v
        for j in range(len(psi)):
            psi_i = np.array([a for a in psi[j] if a < 1])
            for i in range(len(psi_i)):
                wk = optimize.brentq(eq_21, 0, 1, args=psi_i[: i + 1])
                chi_j_k = optimize.brentq(
                    eq_19, 0, 1, args=(self.chi[j], self.price_grid[j][i], self.df[i], wk)
                )
                self.chi[j][i] = chi_j_k
                v_solve = lambda v, k: eq_20(v, k) - chi_j_k
                self.v[j][i] = optimize.brentq(v_solve, 0, 1, args=psi_i[i])

        # main scheme for xi
        self.xi = np.zeros(self.v.shape)
        for i in range(len(self.xi)):
            for j in range(len(self.xi[i])):

                # eq 28 solve for xi
                def v_sq(xi_i_j):
                    self.xi[i][j] = xi_i_j
                    sum = np.sum(
                        [
                            np.sum([self.xi[l][i + k - l] for k in range(j + 1)]) ** 2
                            for l in range(i + 1)
                        ]
                    )
                    return sum - self.v[i][j] ** 2

                if self.v[i][j] > 0:
                    self.xi[i][j] = optimize.brentq(v_sq, 0, 1)

    def get_v(self) -> np.ndarray:
        return self.v

    def get_xi(self) -> np.ndarray:
        return self.xi
