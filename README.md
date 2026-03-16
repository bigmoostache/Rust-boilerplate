## Prior conjugué joint pour une famille exponentielle

### Rappel : famille exponentielle

Une distribution de famille exponentielle s'écrit :

$$p(x \mid \eta) = h(x) \exp\!\Big(\eta^\top T(x) - A(\eta)\Big)$$

où :
- $\eta \in \mathbb{R}^k$ est le **paramètre naturel**
- $T(x)$ est la **statistique suffisante**
- $A(\eta) = \log \int h(x) e^{\eta^\top T(x)} dx$ est la **log-partition** (normalisation)
- $h(x)$ est la mesure de base

---

### Construction du prior conjugué joint

Pour $n$ observations i.i.d., la vraisemblance est :

$$p(\mathbf{x} \mid \eta) = \left(\prod_{i=1}^n h(x_i)\right) \exp\!\Big(\eta^\top \textstyle\sum_i T(x_i) - n A(\eta)\Big)$$

On cherche un prior $p(\eta)$ tel que le posterior soit dans la même famille. La forme conjuguée naturelle est :

$$\boxed{p(\eta \mid \lambda_0, \nu_0) \propto \exp\!\Big(\eta^\top \lambda_0 - \nu_0 A(\eta)\Big)}$$

avec hyperparamètres $(\lambda_0 \in \mathbb{R}^k,\ \nu_0 > 0)$.

---

### Calcul du posterior

Par le théorème de Bayes :

$$p(\eta \mid \mathbf{x}) \propto p(\mathbf{x} \mid \eta)\cdot p(\eta \mid \lambda_0, \nu_0)$$

$$\propto \exp\!\Big(\eta^\top \textstyle\sum_i T(x_i) - n A(\eta)\Big) \cdot \exp\!\Big(\eta^\top \lambda_0 - \nu_0 A(\eta)\Big)$$

$$= \exp\!\left(\eta^\top \underbrace{\left(\lambda_0 + \textstyle\sum_i T(x_i)\right)}_{\lambda_n} - \underbrace{(\nu_0 + n)}_{\nu_n} A(\eta)\right)$$

Le posterior est donc **dans la même famille** avec mise à jour :

$$\boxed{\lambda_n = \lambda_0 + \sum_{i=1}^n T(x_i), \qquad \nu_n = \nu_0 + n}$$

---

### Interprétation des hyperparamètres

| Hyperparamètre | Interprétation |
|---|---|
| $\lambda_0$ | "Somme des statistiques suffisantes" a priori |
| $\nu_0$ | "Nombre d'observations fictives" a priori |
| $\lambda_0 / \nu_0$ | Valeur a priori de $\mathbb{E}[T(x)]$ |

La mise à jour est **additive** : chaque nouvelle observation $x_i$ contribue $T(x_i)$ à $\lambda$ et $1$ à $\nu$. C'est l'expression la plus pure de la conjugaison.

---

### Propriétés structurelles

**1. Le prior est lui-même une famille exponentielle** en $\eta$, avec statistiques suffisantes $(\eta,\ A(\eta))$ et paramètres naturels $(\lambda_0, -\nu_0)$.

**2. Espérance a posteriori.** Puisque $\nabla_\eta A(\eta) = \mathbb{E}_\eta[T(x)]$, on a :

$$\mathbb{E}[\eta \mid \mathbf{x}] \text{ est un shrinkage entre } \frac{\lambda_0}{\nu_0} \text{ et } \frac{\sum_i T(x_i)}{n}$$

Plus précisément, si on note $\bar{T} = \frac{1}{n}\sum_i T(x_i)$ :

$$\frac{\lambda_n}{\nu_n} = \frac{\nu_0}{\nu_0 + n}\cdot\frac{\lambda_0}{\nu_0} + \frac{n}{\nu_0 + n}\cdot\bar{T}$$

C'est une **combinaison convexe** entre prior et données — le prior s'efface quand $n \to \infty$.

**3. Prédictive marginale.** La distribution prédictive s'écrit en fermé :

$$p(x_{n+1} \mid \mathbf{x}) = h(x_{n+1})\frac{Z(\lambda_n + T(x_{n+1}),\ \nu_n + 1)}{Z(\lambda_n, \nu_n)}$$

où $Z(\lambda, \nu)$ est la constante de normalisation du prior conjugué.

---

### En résumé

Pour **toute** famille exponentielle $(h, \eta, T)$, le prior conjugué joint existe, est **universel** dans sa forme $\exp(\eta^\top \lambda_0 - \nu_0 A(\eta))$, et la mise à jour bayésienne se réduit à deux additions — reflétant que $T(x)$ est la statistique suffisante au sens le plus profond.