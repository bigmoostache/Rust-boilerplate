# Modèle Graphique Probabiliste Patient

## 1. Vue d'ensemble

Un champ de Markov (MRF) où chaque nœud $i$ représente une variable clinique portant une **distribution paramétrisée** — pas une valeur ponctuelle. L'inférence produit une mise à jour bayésienne cohérente de l'état du patient, intégrant épidémiologie populationnelle, historique individuel et observations courantes.

---

## 2. Structure du graphe

- **Nœuds** : variables cliniques (glycémie, CRP, compliance, diagnostic, classe de risque…)  
- **Arêtes** : couplages entre variables, paramétrés par des matrices $B_{ij}$  
- **Familles autorisées** : voir section 4 — les lois à un seul paramètre sont interdites

Chaque nœud $i$ porte quatre états successifs :

| Notation | Nom | Description |
|---|---|---|
| $\theta_i^{\text{epidemio}}$ | prior épidémiologique | distribution populationnelle de référence, fixe |
| $\theta_i^{\text{prev}}$ | prior historique | $\theta_i^{\text{post}}$ de l'inférence précédente |
| $\theta_i^{\text{relax}}$ | prior relaxé | $\theta_i^{\text{prev}}$ ramené vers $\theta_i^{\text{epidemio}}$ à vitesse $\tau_i$ |
| $\theta_i^{\text{post}}$ | posterior courant | solution de l'optimisation à l'instant $t$ |

---

## 3. Objectif variationnel

### 3.1 ELBO

On optimise sous hypothèse mean-field $q(\mathbf{x}) = \prod_i p_{\theta_i^{\text{post}}}(x_i)$. L'objectif est :

$$\boxed{\mathcal{F} = \underbrace{\sum_{(i,j)} \mathbb{E}[T_i]^T B_{ij}\, \mathbb{E}[T_j]}_{\text{couplages}} + \underbrace{\sum_i \mathbb{E}\bigl[\log p_{\theta_i^{\text{relax}}}\bigr]}_{\text{priors relaxés}} + \underbrace{\sum_i \sum_{k \in \mathcal{O}_i} \mathbb{E}\bigl[\log p_{\varepsilon_k}\bigr]}_{\text{observations}} + \underbrace{\lambda_S \sum_i \mathbb{H}(p_{\theta_i^{\text{post}}})}_{\text{entropie}}}$$

- $T_i$ : statistiques suffisantes canoniques du nœud $i$ (ex. $(x, x^2)$ pour une Gaussienne)  
- $B_{ij} \in \mathbb{R}^{d_i \times d_j}$ : matrice de couplage entre familles de dimensions éventuellement distinctes  
- $\lambda_S \geq 0$ : échelle globale d'entropie ($\lambda_S = 1$ par défaut)  
- $Z$ disparaît de l'objectif — il ne dépend pas de $\Theta^{\text{post}}$  
- Objectif **entièrement analytique** pour toutes les familles exponentielles

Les quatre termes sont en tension :

| Terme | Rôle |
|---|---|
| Couplages | aligne les $\mathbb{E}[T]$ des nœuds voisins |
| Prior relaxé | ancre vers l'épidémiologie et l'historique patient |
| Observations | met à jour selon les mesures disponibles |
| Entropie ($\times\lambda_S$) | évite le collapse, maintient l'incertitude |

### 3.2 Point fixe

En annulant le gradient naturel $\nabla_{\eta_i}\mathcal{F} = 0$, on obtient le point fixe analytique pour le nœud $i$ :

$$\boxed{\eta_i^* = \frac{\displaystyle\eta_i^{\text{relax}} + \sum_{k \in \mathcal{O}_i} \eta_k^{\text{obs}} + \sum_{j \in \mathcal{N}(i)} B_{ij}\,\mathbb{E}_j[T_j]}{1 + |\mathcal{O}_i| + \lambda_S}}$$

où $\eta_i = \eta(\theta_i^{\text{post}})$ sont les paramètres naturels du nœud $i$, et $D_i = 1 + |\mathcal{O}_i| + \lambda_S$ est le **dénominateur de normalisation** propre à chaque nœud. Le posterior est une moyenne pondérée entre prior relaxé, observations et messages des voisins — l'intuition bayésienne attendue.

---

## 4. Familles autorisées

**Principe** : chaque nœud doit encoder la valeur centrale **et** l'incertitude sur cette valeur. Les lois à un seul paramètre (Bernoulli, Categorical, Poisson) sont interdites — prior et observation y pèsent à égalité quelle que soit la fiabilité de la mesure.

| Variable | Interdit | Autorisé | Pseudo-compte |
|---|---|---|---|
| Continue symétrique | — | Gaussienne $(\mu, \sigma^2)$ | $1/\sigma^2$ |
| Continue $\mathbb{R}^+$ — taux, durées | — | Gamma $(\alpha, \beta)$ | $\beta$ |
| Continue $\mathbb{R}^+$ — biomarqueurs | — | Log-Normale $(\mu, \sigma^2)$ | $1/\sigma^2$ |
| Continue $\mathbb{R}^+$ — queue lourde | — | Inverse-Gamma $(\alpha, \beta)$ | $\beta$ |
| Comptage | Poisson | Gamma $(\alpha, \beta)$ | $\beta$ |
| Discret ($k$ classes) | Bernoulli, Categorical | Dirichlet $\text{Dir}(\boldsymbol{\alpha})$ | $\alpha_0 = \sum_c \alpha_c$ |

Beta$(\alpha_1, \alpha_2)$ est le cas particulier $k=2$ de la Dirichlet.

### Briques analytiques par famille

Pour chaque famille $p_\eta(x) = \exp(\eta^T T(x) - A(\eta) + h(x))$, les quantités utiles pour le point fixe et le jacobien sont :

**Gaussienne** $(\mu, \sigma^2)$

$$\eta = \Bigl(\tfrac{\mu}{\sigma^2},\, -\tfrac{1}{2\sigma^2}\Bigr),\quad T = (x, x^2),\quad \mathbb{E}[T] = (\mu,\, \mu^2+\sigma^2)$$
$$\Lambda = \begin{pmatrix}\sigma^2 & 2\mu\sigma^2 \\ 2\mu\sigma^2 & 2\sigma^4+4\mu^2\sigma^2\end{pmatrix},\quad \mathbb{H} = \tfrac{1}{2}\log(2\pi e\,\sigma^2)$$

**Gamma** $(\alpha, \beta)$

$$\eta = (\alpha-1,\,-\beta),\quad T = (\log x, x),\quad \mathbb{E}[T] = \bigl(\psi(\alpha),\, \tfrac{\alpha}{\beta}\bigr)$$
$$\Lambda = \begin{pmatrix}\psi_1(\alpha) & 0 \\ 0 & \alpha/\beta^2\end{pmatrix} \text{ (diagonale)},\quad \mathbb{H} = \alpha - \log\beta + \log\Gamma(\alpha) + (1-\alpha)\psi(\alpha)$$

**Log-Normale** $(\mu, \sigma^2)$ — identique à Gaussienne sur $\log x$ :

$$T = (\log x, \log^2 x),\quad \mathbb{E}[T] = (\mu, \mu^2+\sigma^2),\quad \mathbb{H} = \mu + \tfrac{1}{2}\log(2\pi e\,\sigma^2)$$

**Inverse-Gamma** $(\alpha, \beta)$, $\alpha > 2$

$$\eta = (-\alpha-1,\,-\beta),\quad T = (\log x, 1/x),\quad \Lambda = \begin{pmatrix}\psi_1(\alpha) & 0 \\ 0 & \beta^2/[(\alpha-1)^2(\alpha-2)]\end{pmatrix}$$

**Dirichlet** $\text{Dir}(\boldsymbol{\alpha})$, $k$ classes

$$\eta = \boldsymbol{\alpha} - \mathbf{1},\quad T = (\log x_1,\ldots,\log x_k),\quad \mathbb{E}[T_c] = \psi(\alpha_c) - \psi(\alpha_0)$$
$$\Lambda = \operatorname{diag}(\psi_1(\boldsymbol{\alpha})) - \psi_1(\alpha_0)\,\mathbf{1}\mathbf{1}^T \quad \text{(rang }k-1\text{, singulière)}$$

$\psi$ : digamma, $\psi_1$ : trigamma. La singularité de $\Lambda$ pour la Dirichlet reflète la contrainte du simplexe : utiliser un pseudo-inverse ou travailler sur les $k-1$ coordonnées libres.

---

## 5. Algorithme d'inférence

### 5.1 Mise à jour d'un nœud

Au lieu de sauter directement sur $\eta_i^*$, on effectue un **pas interpolé** pour éviter les oscillations :

$$\eta_i^{t+1} = (1 - \alpha_i)\,\eta_i^t + \alpha_i\,\eta_i^*\!\left(\eta_{\mathcal{N}(i)}^t\right)$$

La dynamique est contractante :

$$\eta_i^{t+1} - \eta_i^* = (1-\alpha_i)(\eta_i^t - \eta_i^*)$$

### 5.2 Choix de $\alpha_i$

Le jacobien local quantifie l'amplification de perturbation de $j$ vers $i$ :

$$J_{ij} = \frac{\partial \eta_i^*}{\partial \eta_j} = \frac{B_{ij}\,\Lambda_j}{D_i} = \frac{B_{ij}\,\Lambda_j}{1 + |\mathcal{O}_i| + \lambda_S}$$

Le taux de damping adaptatif garantit la contraction :

$$\boxed{\alpha_i = \frac{1}{\max\!\left(1,\;\displaystyle\sum_{j \in \mathcal{N}(i)} \|J_{ij}\|_2\right)}}$$

**Propriétés** :

| Régime | $\sum\|J_{ij}\|_2$ | $\alpha_i$ |
|---|---|---|
| Nœud isolé ou observations fortes | $< 1$ | $1$ — saut direct |
| Couplages modérés | $\sim 1$ | $\sim 0.5$ |
| Graphe dense / $\beta$ forts | $\gg 1$ | $\ll 1$ — très conservatif |

Les observations ($|\mathcal{O}_i|$) et l'entropie ($\lambda_S$) augmentent $D_i$, contractent $\|J_{ij}\|_2$ et permettent mécaniquement des $\alpha_i$ plus grands — la stabilisation est automatique.

---

## 6. Dynamique temporelle

Le prior utilisé à chaque instant $t$ est l'état patient précédent **relaxé** vers l'épidémio :

$$\theta_i^{\text{relax}}(t) = \bigl(1 - e^{-\Delta t/\tau_i}\bigr)\cdot\theta_i^{\text{epidemio}} + e^{-\Delta t/\tau_i}\cdot\theta_i^{\text{prev}}(t)$$

avec $\theta_i^{\text{prev}}(t) = \theta_i^{\text{post}}(t-1)$.

- $\tau_i$ : constante de temps propre à la variable (court pour glycémie, long pour statut fumeur)  
- L'historique est encodé dans $\theta_i^{\text{prev}}$ sans trajectoire explicite  
- Structure de **filtre bayésien hiérarchique** à trois échelles :

| Échelle | Mécanisme | Paramètres |
|---|---|---|
| Épidémio | prior populationnel statique | $\theta_i^{\text{epidemio}}$ |
| Patient | relaxation exponentielle à $\tau_i$ | $\theta_i^{\text{relax}}(t)$ |
| Observation | mise à jour instantanée | $\mathcal{O}_i$ |

---

## 7. Calibration

**Priors épidémiologiques** $\theta_i^{\text{epidemio}}$ — directement depuis la littérature. Justifiables cliniquement.

**Couplages** $B_{ij}$ — calibration en deux étapes :

**Étape 1 — Priors locaux par paire.** Pour chaque arête, priors gaussiens sur les entrées de $B_{ij}$ ($2 \cdot d_i \cdot d_j$ degrés de liberté). L'expert répond à autant de questions cliniques interprétables (espérances conditionnelles). Étape purement locale.

**Étape 2 — Calibration globale par patients virtuels.** Optimisation MAP sur des patients types construits avec l'expert :

$$\hat{B} = \arg\max_{B} \left[\sum_m \log P(\mathbf{x}^{(m)} \mid B) + \log P(B)\right]$$

Les patients types capturent les interactions globales ignorées par l'étape 1.

**Couplages variables** — si $B_{ij}$ semble dépendre du patient, c'est le signal qu'une variable cachée $z_{ij}$ médiatise la relation. On l'ajoute au graphe ; $B_{ij}$ redevient fixe.

---

## 8. Points ouverts

- Choix de la structure du graphe (expert vs. appris)
- Gestion du MNAR (*missing not at random*) dans les EHR
- Validation clinique des posteriors $\theta_i^{\text{post}}$ sur des outcomes réels
- Passage à l'échelle sur des graphes denses