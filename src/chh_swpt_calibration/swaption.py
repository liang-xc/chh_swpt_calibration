import numpy as np
import term_structure
from scipy import stats


class Swaption:
    """
    a class to represent a swaption contract
    """

    def __init__(
        self,
        strike: float,
        maturity: float,
        tenor: float,
        quote: float,
        forward_swap: float,
    ):
        self.strike: float = strike
        self.maturity: float = maturity  # option
        self.tenor: float = tenor  # swap
        self.quote: float = quote  # implied Black vol
        self.forward_swap: float = forward_swap
        self.price: float = self._calc_price()

    def _calc_price(self) -> float:
        """
        a helper function to convert the quote (impolied Black vol) to price

        Returns:
            float: price
        """
        d1 = (
            np.log(self.forward_swap / self.strike)
            + 1 / 2 * self.quote**2 * self.maturity
        ) / (self.quote * np.sqrt(self.maturity))

        d2 = (
            np.log(self.forward_swap / self.strike)
            - 1 / 2 * self.quote**2 * self.maturity
        ) / (self.quote * np.sqrt(self.maturity))

        return self.forward_swap * stats.norm.cdf(d1) - self.strike * stats.norm.cdf(d2)
