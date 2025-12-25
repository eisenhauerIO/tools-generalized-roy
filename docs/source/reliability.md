# Reliability

The following section illustrates the reliability of the estimation strategy behind the `grmpy` package when facing agent heterogeneity and shows also that the corresponding results withstand a critical examination. The checks in both subsections are based on a mock [data set](https://www.aeaweb.org/aer/data/oct2011/20061111_data.zip) respectively the [estimation results](https://assets.aeaweb.org/assets/production/articles-attachments/aer/data/oct2011/20061111_app.pdf#page=9) from:

> Carneiro, Pedro, James J. Heckman, and Edward J. Vytlacil. [Estimating Marginal Returns to Education.](https://pubs.aeaweb.org/doi/pdfplus/10.1257/aer.101.6.2754) *American Economic Review*, 101 (6):2754-81, 2011.

We conduct two different test setups. Firstly we show that `grmpy` is able to provide better results than simple estimation approaches in the presence of essential heterogeneity. Secondly we show that `grmpy` is capable of replicating the $B^{MTE}$ results by {cite}`Carneiro2011` for the parametric version of the Roy model.

## Reliability of Estimation Results

The estimation results and data from {cite}`Carneiro2011` build the basis of the reliability test setup. During each iteration the rate of correlation between the simulated unobservables increases.

For illustrating the reliability we estimate $B^{ATE}$ during each step with two different methods. The first estimation uses a simple OLS approach.

```{figure} figures/fig-ols-average-effect-estimation.png
:align: center
```

As can be seen from the figure, the OLS estimator underestimates the effect significantly. The stronger the correlation between the unobservable variables the more the downwards bias.

```{figure} figures/fig-grmpy-average-effect-estimation.png
:align: center
```

The second figure shows the estimated $B^{ATE}$ from the `grmpy` estimation process. Conversely to the OLS results the estimate of the average effect is close to the true value even if the unobservables are almost perfectly correlated.

## Sensitivity to Different Distributions of the Unobservables

The parametric specification makes the strong assumption that the unobservables follow a joint normal distribution. The semiparametric method of local instrumental variables is more flexible.

### Normal Distribution

```{figure} figures/normal_distribution.png
:align: center
```

Both specifications come very close to the original curve. The parametric model even gets a perfect fit.

### Beta Distribution

The shape of the *beta* distribution can be flexibly adjusted by the tuning parameters $\alpha$ and $\beta$, which we set to 4 and 8, respectively.

```{figure} figures/beta_distribution.png
:align: center
```

The parametric model underestimates the returns to college, whereas the semiparametric $B^{MTE}$ still fits the original curve pretty well.

## Replication

In another check of reliability, we compare the results of our estimation process with already existing results from the literature. We replicate the results for both the parametric and semiparametric MTE from {cite}`Carneiro2011`.

### Parametric Replication

```{figure} figures/fig-marginal-benefit-parametric-replication.png
:align: center
```

### Semiparametric Replication

```{figure} figures/replication_carneiroB.png
:align: center
```

The semiparametric $B^{MTE}$ also gets very close to the original curve.
