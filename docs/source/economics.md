# Economics

This section provides a general discussion of the generalized Roy model and selected issues in the econometrics of policy evaluation.

## Generalized Roy Model

The generalized Roy model ({cite}`HecVyr05`, {cite}`Roy1951`) provides a coherent framework to explore the econometrics of policy evaluation. It is characterized by the following set of equations.

$$
\begin{align}
& \textbf{Potential Outcomes} \\
& Y_1 = \mu_1(X) + U_1 \\
& Y_0 = \mu_0(X) + U_0 \\
& \\
& \textbf{Choice} \\
& D^{*} = \mu_D(Z) - V \\
& D = I[D^{*} > 0] \\
& B = E[Y_1 - Y_0 \mid \mathcal{I}] \\
& \\
& \textbf{Observed Outcome} \\
& Y = D Y_1 + (1 - D) Y_0
\end{align}
$$

$(Y_1, Y_0)$ are objective outcomes associated with each potential treatment state $D$ and realized after the treatment decision. $Y_1$ refers to the outcome in the treated state and $Y_0$ in the untreated state. $D^{*}$ denotes the latent tendency for treatment participation. Agents take up treatment $D$ if their latent tendency $D^{*}$ is positive. $\mathcal{I}$ denotes the agent's information set at the time of the participation decision.

From the perspective of the econometrician, $(X, Z)$ are observable while $(U_1, U_0, V)$ are not. $X$ are the observed determinants of potential outcomes $(Y_1, Y_0)$, and $Z$ are the observed determinants of the latent indicator variable $D^{*}$.

## Agent Heterogeneity

What gives rise to variation in choices and outcomes among, from the econometrician's perspective, otherwise observationally identical agents? This is the central question in all econometric policy analyses ({cite}`BrHecHa07`, {cite}`Heckman2001`).

The individual benefit of treatment is defined as

$$
B = Y_1 - Y_0 = (\mu_1(X) - \mu_0(X)) + (U_1 - U_0).
$$

The concept of essential heterogeneity emphasizes that if agents select their treatment status based on benefits unobserved by the econometrician (selection on unobservables), then there is no unique effect of a treatment or a policy even after conditioning on observable characteristics.

## Objects of Interest

Treatment effect heterogeneity requires us to be precise about the effect being discussed.

### Conventional Average Treatment Effects

It is common to summarize the average benefits of treatment for different subsets of the population:

$$
B^{ATE} = E[Y_1 - Y_0] \\
B^{TT} = E[Y_1 - Y_0 | D = 1] \\
B^{TUT} = E[Y_1 - Y_0 | D = 0]
$$

```{figure} figures/fig-treatment-effects-with-and-without-eh.png
:align: center

**Fig. 1:** Conventional treatment effects with and without essential heterogeneity
```

### Policy-Relevant Average Treatment Effect

The $B^{PRTE}$ captures the average change in outcomes per net person shifted by a change from a baseline state $B$ to an alternative policy $A$:

$$
B^{PRTE} = \frac{1}{E[D_A] - E[D_B]}(E[Y_A] - E[Y_B])
$$

### Local Average Treatment Effect

The Local Average Treatment Effect $B^{LATE}$ was introduced by {cite}`Imbens94`. They show that instrumental variable approaches (IV) identify $B^{LATE}$.

```{figure} figures/fig-local-average-treatment.png
:align: center

**Fig. 2:** $B^{LATE}$ at different values of $u_S$
```

### Marginal Treatment Effect

The $B^{MTE}$ is the treatment effect parameter that conditions on the unobserved desire to select into treatment:

$$
B^{MTE}(u_S) = E[Y_1 - Y_0 | U_S = u_S]
$$

The $B^{MTE}$ provides the underlying structure for all average effect parameters. These can be derived as weighted averages of the $B^{MTE}$:

$$
\Delta j = \int_{0}^{1} B^{MTE}(u_S) \omega^{j}(u_S) du_S
$$

```{figure} figures/fig-weights-marginal-effect.png
:align: center

**Fig. 3:** Weights for the marginal treatment effect for different parameters.
```

```{figure} figures/fig-eh-marginal-effect.png
:align: center

**Fig 4:** $B^{MTE}$ in the presence and absence of essential heterogeneity.
```

## Distribution of Potential Outcomes

Several interesting aspects of policies cannot be evaluated without knowing the joint distribution of potential outcomes ({cite}`AbbHec07`). The joint distribution of $(Y_1, Y_0)$ allows calculating the whole distribution of benefits.

```{figure} figures/fig-distribution-joint-potential.png
:align: center

**Fig 5:** Distribution of potential outcomes
```

```{figure} figures/fig-distribution-joint-surplus.png
:align: center

**Fig. 6:** Distribution of benefits and surplus
```
