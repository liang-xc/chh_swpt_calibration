import numpy as np
from scipy import interpolate


class TermStructure:
    def __init__(self, t: np.ndarray):
        self.time_grid: np.ndarray = t


class YieldCurve(TermStructure):
    """
    a class to represent a yield curve
    """

    def __init__(self, t: np.ndarray, rates: np.ndarray):
        assert len(t) == len(rates), "time grid and rates should have same length"
        super().__init__(t)
        self.rates: np.ndarray = rates

    def __call__(
        self, t: float, method: str = "previous", extrapolate: bool = True
    ) -> float:
        """
        yield rate at time to maturity t

        Args:
            t (float): time to maturity
            method (str, optional): interpolation method. Defaults to "previous".
            extrapolate (bool, optional): whether to extrapolate. Defaults to True.

        Returns:
            float: yield rate
        """
        if extrapolate:
            return interpolate.interp1d(
                self.time_grid, self.rates, kind=method, fill_value="extrapolate"
            )(t)
        else:
            return interpolate.interp1d(self.time_grid, self.rates, kind=method)(t)

    def to_discount_curve(self) -> "DiscountCurve":
        """
        convert the yield curve to a discount curve

        Returns:
            DiscountCurve: discount curve
        """
        return DiscountCurve(
            self.time_grid,
            np.array([np.exp(-r * t) for t, r in zip(self.time_grid, self.rates)]),
        )

    def to_forward_curve(self) -> "ForwardCurve":
        """
        convert the yield curve to a forward curve

        Returns:
            ForwardCurve: forward curve
        """
        f = np.zeros(len(self.time_grid))
        f[0] = self.rates[0]
        for i in range(1, len(self.time_grid)):
            f[i] = (
                self.rates[i] * self.time_grid[i]
                - self.rates[i - 1] * self.time_grid[i - 1]
            ) / (self.time_grid[i] - self.time_grid[i - 1])
        return ForwardCurve(self.time_grid, f)


class DiscountCurve(TermStructure):
    """
    a class to represent a discount curve
    """

    def __init__(self, t: np.ndarray, df: np.ndarray):
        assert len(t) == len(df), "number of discount factors does not match time"
        super().__init__(t)
        self.df: np.ndarray = df

    def __call__(
        self, t: float, method: str = "previous", extrapolate: bool = True
    ) -> float:
        y = self.to_yield_curve()
        r = y(t, method, extrapolate)
        return np.exp(r * -t)

    def to_yield_curve(self) -> YieldCurve:
        """
        convert the discount curve to a yield curve

        Returns:
            YieldCurve: yield curve
        """
        return YieldCurve(
            self.time_grid,
            np.array([-np.log(d) / t for t, d in zip(self.time_grid, self.df)]),
        )


class ForwardCurve(TermStructure):
    """
    a class to represent a forward curve observed at t = 0
    """

    def __init__(self, t: np.ndarray, forward_rates: np.ndarray):
        assert len(t) == len(forward_rates), "number of forawrd rates does not match time"
        super().__init__(t)
        self.foward = forward_rates

    def __call__(
        self, t1: float, t2: float, method: str = "previous", extrapolate: bool = True
    ) -> float:
        if t1 > t2:
            t1, t2 = t2, t1
        y = self.to_yield_curve()
        r1 = y(t1, method, extrapolate)
        r2 = y(t2, method, extrapolate)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def to_yield_curve(self) -> YieldCurve:
        """
        convert the forward curve to a yield curve

        Returns:
            YieldCurve: yield curve
        """
        r = np.zeros(len(self.time_grid))
        r[0] = self.forward[0]
        for i in range(1, len(r)):
            r[i] = (
                r[i - 1] * self.time_grid[i - 1]
                + self.forward[i] * (self.time_grid[i] - self.time_grid[i - 1])
            ) / self.time_grid[i]
        return YieldCurve(self.time_grid, r)
