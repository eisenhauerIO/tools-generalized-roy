# Tutorial

We now illustrate the basic capabilities of the `grmpy` package. We start by outlining some basic functional form assumptions before introducing alternative models that can be used to estimate the marginal treatment effect (MTE). We then turn to some simple use cases.

## Assumptions

The `grmpy` package implements the normal linear-in-parameters version of the generalized Roy model. Both potential outcomes and the choice $(Y_1, Y_0, D)$ are a linear function of the individual's observables $(X, Z)$ and random components $(U_1, U_0, V)$.

$$
Y_1 = X \beta_1 + U_1 \\
Y_0 = X \beta_0 + U_0 \\
D = I[D^{*} > 0] \\
D^{*} = Z \gamma - V
$$

Individuals decide to select into treatment if the latent indicator variable $D^{*}$ is positive. Depending on their decision, we either observe $Y_1$ or $Y_0$.

### Parametric Normal Model

The parametric model imposes the assumption of joint normality of the unobservables $(U_1, U_0, V) \sim \mathcal{N}(0, \Sigma)$ with mean zero and covariance matrix $\Sigma$.

### Semiparametric Model

The semiparametric approach invokes no assumption on the distribution of the unobservables. It requires a weaker condition $(X,Z) \perp (U_1, U_0, V)$.

Under this assumption, the MTE is:

* additively separable in $X$ and $U_D$, which means that the shape of the MTE is independent of $X$, and
* identified over the common support of $P(Z)$, unconditional on $X$.

The assumption of common support is crucial for the application of LIV and needs to be carefully evaluated every time. It is defined as the region where the support of $P(Z)$ given $D=1$ and the support of $P(Z)$ given $D=0$ overlap.

## Model Specification

You can specify the details of the model in an initialization file ([example](https://github.com/OpenSourceEconomics/grmpy/blob/master/docs/tutorial/tutorial.grmpy.yml)). This file contains several blocks:

### SIMULATION

The *SIMULATION* block contains some basic information about the simulation request.

| Key | Value | Interpretation |
|-----|-------|----------------|
| agents | int | number of individuals |
| seed | int | seed for the specific simulation |
| source | str | specified name for the simulation output files |

### ESTIMATION

Depending on the model, different input parameters are required.

**Parametric Model**

| Key | Value | Interpretation |
|-----|-------|----------------|
| semipar | False | choose the parametric normal model |
| agents | int | number of individuals (for the comparison file) |
| file | str | name of the estimation specific init file |
| optimizer | str | optimizer used for the estimation process |
| start | str | flag for the start values |
| maxiter | int | maximum numbers of iterations |
| dependent | str | indicates the dependent variable |
| indicator | str | label of the treatment indicator variable |
| output_file | str | name for the estimation output file |
| comparison | int | flag for enabling the comparison file creation |

**Semiparametric Model**

| Key | Value | Interpretation |
|-----|-------|----------------|
| semipar | True | choose the semiparametric model |
| show_output | bool | If *True*, intermediate outputs of the estimation process are displayed |
| dependent | str | indicates the dependent variable |
| indicator | str | label of the treatment indicator variable |
| file | str | name of the estimation specific init file |
| logit | bool | If false: probit. Probability model for the choice equation |
| nbins | int | Number of histogram bins used to determine common support (default is 25) |
| bandwidth | float | Bandwidth for the locally quadratic regression |
| gridsize | int | Number of evaluation points for the locally quadratic regression (default is 400) |
| ps_range | list | Start and end point of the range of $p = u_D$ over which the MTE shall be estimated |
| rbandwidth | int | Bandwidth for the double residual regression (default is 0.05) |
| trim_support | bool | Trim the data outside the common support, recommended (default is *True*) |
| reestimate_p | bool | Re-estimate $P(Z)$ after trimming, not recommended (default is *False*) |

In most empirical applications, bandwidth choices between 0.2 and 0.4 are appropriate. {cite}`Fan1994` find that a gridsize of 400 is a good default for graphical analysis.

### TREATED

The *TREATED* block specifies the number and order of the covariates determining the potential outcome in the treated state and the values for the coefficients $\beta_1$.

| Key | Container | Values | Interpretation |
|-----|-----------|--------|----------------|
| params | list | float | Parameters |
| order | list | str | Variable labels |

### UNTREATED

The *UNTREATED* block specifies the covariates that affect the potential outcome in the untreated state and the values for the coefficients $\beta_0$.

| Key | Container | Values | Interpretation |
|-----|-----------|--------|----------------|
| params | list | float | Parameters |
| order | list | str | Variable labels |

### CHOICE

The *CHOICE* block specifies the number and order of the covariates determining the selection process and the values for the coefficients $\gamma$.

| Key | Container | Values | Interpretation |
|-----|-----------|--------|----------------|
| params | list | float | Parameters |
| order | list | str | Variable labels |

### DIST (Parametric Model Only)

The *DIST* block specifies the distribution of the unobservables.

| Key | Container | Values | Interpretation |
|-----|-----------|--------|----------------|
| params | list | float | Upper triangular of the covariance matrix |

## Examples

### Parametric Normal Model

**Simulation**

For simulating a sample from the generalized Roy model you use the `simulate()` function provided by the package:

```python
import grmpy

grmpy.simulate('tutorial.grmpy.yml')
```

This creates a number of output files that contain information about the resulting simulated sample:

* **data.grmpy.info** - basic information about the simulated sample
* **data.grmpy.txt** - simulated sample in a simple text file
* **data.grmpy.pkl** - simulated sample as a pandas data frame

**Estimation**

The other feature of the package is the estimation of the parameters of interest. By default, the parametric model is chosen:

```python
grmpy.fit('tutorial.grmpy.yml', semipar=False)
```

### Local Instrumental Variables

If the user wishes to estimate the parameters of interest using the semiparametric LIV approach, *semipar* must be changed to *True*:

```python
grmpy.fit('tutorial.semipar.yml', semipar=True)
```

If *show_output* is *True*, `grmpy` plots the common support of the propensity score and shows some intermediate outputs of the estimation process.
