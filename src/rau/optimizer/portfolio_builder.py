import numpy as np
import pandas as pd
import inspect
import warnings
from scipy.optimize import minimize, Bounds
from copy import deepcopy
from typing import List, Dict, Callable, Union, Optional, Tuple


class OptPortfolioBuilder:
    """
    Conduct numerical optimization to obtain optimal allocation weight.
    
    APIs
    -------
    * `build_optimal_weight()`
    * `build_optimal_rolling_weight()`

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> num_assets = 5
    >>> returns = pd.DataFrame(np.random.normal(0.05, 0.3, (100, num_assets)))

    >>> # Use built-in objective function
    >>> optimizer = OptPortfolioBuilder(objective="variance", minimize=True)
    >>> weight = optimizer.build_optimal_weight(returns)

    >>> # Use user-defined objective function
    >>> my_objective = lambda weights, returns: (weights.reshape(1, -1) * returns).mean()
    >>> optimizer = OptPortfolioBuilder(objective=my_objective, minimize=False)
    >>> weight = optimizer.build_optimal_weight(returns)

    >>> # Rolling optimization
    >>> optimizer = OptPortfolioBuilder()
    >>> weights = optimizer.build_rolling_optimal_weight(returns=returns, calib_freq="1Q", lookback_length="365d")
    """

    BUILTIN_OBJECTIVES = [
        'raw_sharpe',
        'variance',
    ]

    def __init__(
        self,
        objective: Union[str, Callable] = "raw_sharpe",
        minimize: bool = False,
        bounds: List[float] = [0, 1],
        constraints: Tuple[Dict] = (),
        method: str = 'SLSQP',
        options: Optional[Dict] = None,
    ):
        """

        Args:
            objective (Union[str, Callable], optional): Optimization target function; `fucntion(weights, returns) -> float`.
            minimize (bool, optional): Defaults to False.
            bounds (List[float], optional): Lower/upper bound of allocation weights. Defaults to [0, 1].
            constraints (Tuple[Dict], optional): Additional constraints. Input format comply with `scipy.minimize()`.
            method (str, optional): Defaults to 'SLSQP'. Available methods are equivalent to `scipy.minimize()`.
            options (Optional[Dict], optional):
        """
        self.minimize = minimize
        self.objective = self._build_objective_func(objective)
        self.bounds = Bounds(*bounds)
        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, ) + constraints
        self.method = method
        self.options = options
    
    def build_optimal_weight(
        self,
        returns: pd.DataFrame,
        initial_weight: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """

        Args:
            returns (pd.DataFrame): Realized return in `pandas.DataFrame` with shape of [T x #Assets].
            initial_weight (Optional[np.ndarray], optional): Defaults to equal weighting.

        Returns:
            pd.Series:
        """

        # set initial weight
        if initial_weight is None:
            initial_weight = np.ones(returns.shape[1]) / returns.shape[1]
        
        # run optimization
        result = minimize(
            self.objective,
            initial_weight,
            args=(returns.values,),
            method=self.method,
            bounds=self.bounds, 
            constraints=self.constraints,
            options=self.options,
        )

        # extract optimization result
        asset_names = returns.columns
        optimal_weights = pd.Series(result.x, index=asset_names)
        return optimal_weights

    def build_rolling_optimal_weight(
        self,
        returns: pd.DataFrame,
        calib_freq: str = "1Q",
        lookback_length: str = "365d",
        initial_weight: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """

        Args:
            returns (pd.DataFrame):
            calib_freq (str, optional): Defaults to "1Q".
            lookback_length (str, optional): Defaults to "365d".
            initial_weight (Optional[np.ndarray], optional): Defaults to None.

        Returns:
            pd.DataFrame:
        """
        assert isinstance(returns.index, pd.DatetimeIndex), "`returns` must be indexed by `pd.DateTimeIndex`."
        
        # run rolling optimization
        optimal_weights = []
        time_index = returns.index.rename("datetime")
        calib_time_index = time_index.to_frame().resample(calib_freq).last().iloc[:, 0]
        for t in calib_time_index:
            ret = returns.loc[t-pd.Timedelta(lookback_length):t-pd.Timedelta("1s")]  # do not include today's return
            if ret.shape[0] < 10:
                warnings.warn(category=RuntimeWarning, message=f"#sample<10, skip calibration at t={t}")
                continue
            weight = self.build_optimal_weight(returns=ret, initial_weight=initial_weight)
            weight.name = t
            optimal_weights.append(weight)
        
        # tidy up output weight timeseries
        optimal_weights = pd.DataFrame(optimal_weights)  # concat in one DataFrame
        optimal_weights = optimal_weights.asof(returns.index)  # reindex
        return optimal_weights

    def _build_objective_func(self, objective: Union[str, Callable]) -> Callable:
        # Build optimization target function in required format
        multiplier = 2 * self.minimize - 1

        # Built-in function
        if isinstance(objective, str):
            if objective == "raw_sharpe":
                return lambda weights, returns: multiplier * OptPortfolioBuilder._raw_sharpe(weights, returns)
            elif objective == "variance":
                return lambda weights, returns: multiplier * OptPortfolioBuilder._variance(weights, returns)
            elif objective not in self.BUILTIN_OBJECTIVES:
                raise NotImplementedError(f"`objective` {objective} is not implement as built-in method.")
        
        # User-defined objective function
        elif isinstance(objective, Callable):
            assert OptPortfolioBuilder._get_positional_argnames(objective) == ["weights", "returns"],\
                "`objective` function must be `func(weights, returns)`."
            return lambda weights, returns: multiplier * objective(weights, returns)
        
        else:
            raise TypeError("`objective` must be type of str of callable.")
    
    @staticmethod
    def _raw_sharpe(weights: np.ndarray, returns: np.ndarray) -> float:
        # portfolio realized sharpe
        weighted_returns = (weights.reshape(1, -1) * returns).sum(axis=1)
        return weighted_returns.mean() / weighted_returns.std()

    @staticmethod
    def _variance(weights: np.ndarray, returns: np.ndarray) -> float:
        # portfolio realized return volatility
        weighted_returns = (weights.reshape(1, -1) * returns).sum(axis=1)
        return weighted_returns.std()

    @staticmethod
    def _get_positional_argnames(func: Callable) -> List[str]:
        # get argument name list
        positional_args = [
            param.name for param in inspect.signature(func).parameters.values()
            if param.default == inspect.Parameter.empty 
            and param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return positional_args
