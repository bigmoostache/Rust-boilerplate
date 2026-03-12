## Loi conditionnelle dans un modèle exponentiel couplé

### Modèle joint

On part de la distribution jointe définie par :

$$p(x_1, x_2) \propto \exp\!\Big(\eta_1^\top T_1(x_1) + \eta_2^\top T_2(x_2) + T_1(x_1)^\top B\, T_2(x_2)\Big)$$

où $B$ est la matrice de couplage, et $T_1, T_2$ sont les statistiques suffisantes de chaque famille.

---

### Conditionnement par rapport à $x_2$

On fixe $x_2$ et on regarde ce qui dépend de $x_1$ :

$$p(x_1 \mid x_2) \propto \exp\!\Big(\eta_1^\top T_1(x_1) + T_1(x_1)^\top B\, T_2(x_2)\Big)$$

On regroupe les deux termes en $T_1(x_1)$ :

$$\boxed{p(x_1 \mid x_2) \propto \exp\!\Big(\underbrace{\left(\eta_1 + B\, T_2(x_2)\right)}_{\eta_1^{\text{new}}(x_2)}{}^\top T_1(x_1)\Big)}$$

---

### Résultat clé

La loi conditionnelle $p(x_1 \mid x_2)$ est dans **la même famille exponentielle que $x_1$**, avec un paramètre naturel mis à jour :

$$\eta_1^{\text{new}}(x_2) = \eta_1 + B\, T_2(x_2)$$

Autrement dit, $x_2$ **n'entre que via sa statistique suffisante** $T_2(x_2)$, et l'effet du couplage est un simple **décalage affine** dans l'espace des paramètres naturels.

---

### Utilisation pour la calibration.

Supposons que je veuille calibrer l'état entre $X_1$ le noeud température (gaussien), et $X_2$ le noeud grippe (beta).
La calibration locale se fait par l'expression de "statements" tels que
- E[X_1|X_2=1]+39.5
- V[X_1|X_2=1] ne change pas
- eta_1|X_2=0 = eta_1 (ça bouge pas)


## Calibration du couplage Gaussien–Beta

### Paramétrisations des deux familles

**Gaussienne** $X_1 \sim \mathcal{N}(\mu, \sigma^2)$ :
$$T_1(x_1) = \begin{pmatrix} x_1 \\ x_1^2 \end{pmatrix}, \quad \eta_1 = \begin{pmatrix} \mu/\sigma^2 \\ -1/(2\sigma^2) \end{pmatrix}$$

**Beta** $X_2 \sim \text{Beta}(\alpha, \beta)$ :
$$T_2(x_2) = \begin{pmatrix} \log x_2 \\ \log(1-x_2) \end{pmatrix}$$

**Matrice de couplage** à déterminer :
$$B = \begin{pmatrix} b_{11} & b_{12} \\ b_{21} & b_{22} \end{pmatrix}$$

Le paramètre conditionnel est $\eta_1^{\text{new}}(x_2) = \eta_1 + B\, T_2(x_2)$, ce qui donne des moments conditionnels :

$$\mu(x_2) = -\frac{\eta_{1,1} + b_{11}\log x_2 + b_{12}\log(1-x_2)}{2\left(\eta_{1,2} + b_{21}\log x_2 + b_{22}\log(1-x_2)\right)}$$

$$\sigma^2(x_2) = -\frac{1}{2\left(\eta_{1,2} + b_{21}\log x_2 + b_{22}\log(1-x_2)\right)}$$

---

### Contrainte 1 : $\eta_1 \mid X_2 = 0$ ne bouge pas

On veut $B \cdot T_2(0) = 0$, avec $T_2(0) = (\log 0,\ \log 1)^\top = (-\infty,\ 0)^\top$.

Pour que le résultat soit fini et nul, il faut impérativement :
$$b_{11} = 0, \quad b_{21} = 0$$

Sinon on aurait un divergence $b_{i1} \cdot (-\infty)$. La matrice se réduit à :

$$B = \begin{pmatrix} 0 & b_{12} \\ 0 & b_{22} \end{pmatrix}$$

---

### Contrainte 2 : $\mathbb{V}[X_1 \mid X_2 = 0.99]$ ne change pas

La variance conditionnelle est $\sigma^2(x_2) = -1/(2\eta_{1,2}^{\text{new}})$, donc elle est inchangée ssi $\eta_{1,2}^{\text{new}} = \eta_{1,2}$, i.e. :

$$b_{22} \cdot \log(1 - 0.99) = 0 \implies b_{22} \cdot \log(0.01) = 0$$

Comme $\log(0.01) \neq 0$, on conclut :
$$b_{22} = 0$$

La matrice se réduit à :
$$B = \begin{pmatrix} 0 & b_{12} \\ 0 & 0 \end{pmatrix}$$

---

### Contrainte 3 : $\mathbb{E}[X_1 \mid X_2 = 0.99] = 39.5$

Avec $B$ simplifié, le paramètre conditionnel est :
$$\eta_{1,1}^{\text{new}} = \eta_{1,1} + b_{12} \cdot \log(0.01), \qquad \eta_{1,2}^{\text{new}} = \eta_{1,2}$$

La moyenne conditionnelle vaut :
$$\mu(0.99) = -\frac{\eta_{1,1} + b_{12}\log(0.01)}{2\eta_{1,2}} = \mu - b_{12} \cdot \underbrace{\frac{\log(0.01)}{2\eta_{1,2}}}_{= \log(0.01)\cdot\sigma^2 \cdot (-1)}$$

$$\mu(0.99) = \mu + b_{12} \cdot |\log(0.01)| \cdot \sigma^2$$

En posant la contrainte $\mu(0.99) = 39.5$ :

$$\boxed{b_{12} = \frac{39.5 - \mu}{\log(0.01)\cdot \sigma^2} = \frac{39.5 - \mu}{-\log(100)\cdot \sigma^2}}$$

---

### Résultat final

$$B = \begin{pmatrix} 0 & \dfrac{39.5 - \mu}{\log(0.01)\cdot \sigma^2} \\[8pt] 0 & 0 \end{pmatrix}$$

Le couplage agit **uniquement via $\log(1-x_2)$**, qui encode la "santé" (proche de $\log 1 = 0$ si sain, fortement négatif si $x_2 \to 1$). La structure est intuitive :

| $x_2$ (proba grippe) | $\log(1-x_2)$ | effet sur $\mu$ |
|---|---|---|
| $\approx 0$ (sain) | $\approx 0$ | aucun |
| $0.5$ | $-0.69$ | légère hausse |
| $0.99$ (grippe) | $-4.6$ | $+39.5 - \mu$ |