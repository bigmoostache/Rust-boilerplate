# Modèle Graphique Probabiliste Patient

## Vue d'ensemble

Un graphe de Markov (MRF) où chaque nœud représente une variable clinique, portant une distribution paramétrisée. L'inférence produit une mise à jour bayésienne cohérente de l'état du patient, intégrant épidémiologie, historique et observations.

---

## Structure du graphe

- **Nœuds** : variables cliniques (glycémie, CRP, compliance, diagnostic, classe de risque...)
- **Arêtes** : couplages entre variables, paramétrés par $\beta_{ij}$
- **Familles supportées** : Gaussienne, Gamma, Beta, Poisson, Bernoulli, Categorical, Dirichlet

Chaque nœud $i$ porte quatre états successifs :

| Notation | Nom | Description |
|---|---|---|
| $\theta_i^{\text{epidemio}}$ | prior non informé | distribution populationnelle de référence, fixe |
| $\theta_i^{\text{prev}}$ | prior non relaxé | état patient issu de l'inférence précédente |
| $\theta_i^{\text{relax}}$ | prior relaxé | $\theta_i^{\text{prev}}$ ramené vers $\theta_i^{\text{epidemio}}$ à vitesse $\tau_i$ |
| $\theta_i^{\text{post}}$ | state / posterior | état courant, solution de l'optimisation |

Le posterior est défini comme :

$$\boldsymbol{\theta}^{\text{post}} = \arg\max_{\Theta} \, \mathcal{F}(\mathbf{x}, \Theta^{\text{relax}}, \boldsymbol{\beta}, \mathcal{O})$$

---

## Objectif variationnel (ELBO)

### Étape 1 — Cas scalaire : le MRF ponctuel

Si chaque nœud portait une valeur ponctuelle $x_i \in \mathbb{R}$ (cas Ising généralisé), la log-probabilité jointe s'écrirait naturellement :

$$\log P(\mathbf{x}) = \sum_i \log p_{\theta_i^{\text{relax}}}(x_i) + \sum_{(i,j)} \beta_{ij} \, T(x_i) \cdot T(x_j) - \log Z$$

- le premier terme est la vraisemblance locale de chaque valeur sous son prior relaxé
- le second est le couplage à la Ising, généralisé via les statistiques suffisantes $T$
- $\log Z$ est la constante de normalisation

### Étape 2 — Passage aux distributions

On ne travaille pas sur des valeurs ponctuelles mais sur des distributions $p_{\theta_i^{\text{post}}}$. On évalue donc $\log P$ **en moyenne** sous la distribution jointe factorisée $q(\mathbf{x}) = \prod_i p_{\theta_i^{\text{post}}}(x_i)$ — hypothèse mean field, justifiée ici par la contrainte structurelle de famille paramétrique fixée sur chaque nœud :

$$\mathbb{E}_q\left[\log P(\mathbf{x})\right] = \sum_i \mathbb{E}_{\theta_i^{\text{post}}}\left[\log p_{\theta_i^{\text{relax}}}(x_i)\right] + \sum_{(i,j)} \beta_{ij} \cdot \mathbb{E}_{\theta_i^{\text{post}}}[T(x_i)] \cdot \mathbb{E}_{\theta_j^{\text{post}}}[T(x_j)] - \log Z$$

Le terme de couplage factorise car $x_i$ et $x_j$ sont indépendants sous $q$.

### Étape 3 — $Z$ disparaît

$Z$ dépend de $\Theta^{\text{relax}}$ et $\boldsymbol{\beta}$, mais **pas de $\Theta^{\text{post}}$**. C'est une constante pour l'optimisation — elle disparaît.

### Étape 4 — Ajout de l'entropie

$\mathbb{E}_q[\log P]$ seul pousse $q$ à coller au mode de $P$ — collapse vers une distribution dégénérée (variance nulle). On ajoute $\mathbb{H}(q)$ pour obtenir l'ELBO complet, objectif bayésien canonique qui équilibre fidélité au modèle et maintien de l'incertitude.

### Étape 5 — Observations $\mathcal{O}_i$

Les observations s'ajoutent comme des potentiels nodaux supplémentaires dans $\log P$, indépendants du prior. Chaque nœud peut recevoir zéro, une ou plusieurs observations — elles s'accumulent par somme de log-vraisemblances.

### Résultat

$$\boxed{\mathcal{F}(\Theta^{\text{post}}, \Theta^{\text{relax}}, \boldsymbol{\beta}, \mathcal{O}) = \underbrace{\sum_{(i,j)} \mathbb{E}_{\theta_i^{\text{post}}}[T_i]^T \, B_{ij} \, \mathbb{E}_{\theta_j^{\text{post}}}[T_j]}_{\text{couplages}} + \underbrace{\sum_i \mathbb{E}_{\theta_i^{\text{post}}}[\log p_{\theta_i^{\text{relax}}}]}_{\text{priors relaxés}} + \underbrace{\sum_i \sum_{k \in \mathcal{O}_i} \mathbb{E}_{\theta_i^{\text{post}}}[\log p_{\varepsilon_k}]}_{\text{observations}} + \underbrace{\sum_i \mathbb{H}(p_{\theta_i^{\text{post}}})}_{\text{entropie}}}$$

- **Entièrement analytique** pour toutes les familles exponentielles
- $Z$ éliminé — pas d'intractabilité résiduelle dans l'objectif
- Les nœuds sans observation sont imputés naturellement via les couplages
- Les $T_i$ sont les **statistiques suffisantes** canoniques de chaque famille (ex: $(\mu, \mu^2+\sigma^2)$ pour une Gaussienne)

---

## Construction des termes de couplage

### Dérivation depuis Ising

Dans le modèle d'Ising original, la **log-probabilité jointe** sur des spins $s_i \in \{-1, +1\}$ s'écrit :

$$\log P(\mathbf{s}) = \sum_{(i,j)} \beta_{ij} \, s_i \, s_j + \sum_i h_i \, s_i - \log Z$$

Le terme $\beta_{ij} \, s_i \, s_j$ est donc une contribution à la **log-prob** — pas à la prob. Il module l'énergie du système : configurations où $s_i$ et $s_j$ sont alignés sont plus ou moins probables selon le signe de $\beta_{ij}$.

La généralisation à des variables continues $x_i \in \mathbb{R}$ est directe : on remplace $s_i$ par une statistique $h(x_i)$ de la variable. La log-prob jointe devient :

$$\log P(\mathbf{x}) = \sum_i \log p_{\theta_i^{\text{relax}}}(x_i) + \sum_{(i,j)} \beta_{ij} \cdot h(x_i) \cdot h(x_j) - \log Z$$

Le signe de $\beta_{ij}$ contrôle la nature du couplage : $\beta_{ij} > 0$ favorise l'alignement des valeurs de $h$, $\beta_{ij} < 0$ les oppose. La question qui reste est : quel choix de $h$ ?

### Choix de $h$ : pourquoi les statistiques suffisantes

Le choix de $h$ n'est pas anodin. Pour une famille exponentielle, la distribution s'écrit :

$$p_\theta(x) = \exp\left(\eta(\theta)^T T(x) - A(\theta)\right)$$

Choisir $h = T$ — les statistiques suffisantes — est optimal pour trois raisons :

**1. Exhaustivité** — $T(x)$ capture toute l'information de $x$ sur $\theta$. Toute autre fonction $h$ perd de l'information.

**2. Analyticité** — $\mathbb{E}_\theta[T(x)] = \nabla_\eta A(\eta)$, calculable en forme fermée pour toutes les familles exponentielles. C'est ce qui garantit l'analyticité de l'objectif.

**3. Cohérence géométrique** — le couplage opère dans l'espace naturel de la famille, pas dans un espace arbitraire.

Les alternatives courantes et leurs compromis :

| Choix de $h$ | Avantage | Inconvénient |
|---|---|---|
| $T(x)$ statistiques suffisantes | exhaustif, analytique | dimension variable selon la famille |
| $x$ seul (moment brut) | simple, universel | perd l'information sur la variance |
| $\eta(\theta)$ paramètres naturels | linéaire dans l'espace naturel | moins interprétable cliniquement |
| $\mathbb{E}[x]$ seul | très simple | ignore toute l'incertitude |

### Couplages inter-familles et structure matricielle

Quand les nœuds $i$ et $j$ appartiennent à des familles différentes, leurs statistiques suffisantes $T_i$ et $T_j$ peuvent avoir des dimensions différentes :

- $T_i \in \mathbb{R}^{d_i}$ — par exemple $d_i = 2$ pour une Gaussienne : $(x, x^2)$
- $T_j \in \mathbb{R}^{d_j}$ — par exemple $d_j = k$ pour une Categorical : $(\mathbf{1}_{x=1}, \ldots, \mathbf{1}_{x=k})$

Le couplage scalaire $\beta_{ij} \in \mathbb{R}$ ne suffit plus. On introduit une **matrice de couplage** $B_{ij} \in \mathbb{R}^{d_i \times d_j}$ :

$$\psi_{ij}(x_i, x_j) = \exp\left(T_i(x_i)^T \, B_{ij} \, T_j(x_j)\right)$$

Ce qui donne après passage en espérance sous $q$ :

$$\mathbb{E}_q\left[T_i^T B_{ij} T_j\right] = \mathbb{E}_{\theta_i^{\text{post}}}[T_i]^T \, B_{ij} \, \mathbb{E}_{\theta_j^{\text{post}}}[T_j]$$

Chaque entrée $(a, b)$ de $B_{ij}$ contrôle l'interaction entre la $a$-ème statistique de $i$ et la $b$-ème statistique de $j$. Par exemple pour un couplage Gamma ($d=2$) — Categorical ($d=k$) :

$$B_{ij} = \begin{pmatrix} b_{11} & \cdots & b_{1k} \\ b_{21} & \cdots & b_{2k} \end{pmatrix}$$

où $b_{1c}$ couple $\mathbb{E}[\log x_i]$ à la probabilité de classe $c$, et $b_{2c}$ couple $\mathbb{E}[x_i]$ à la même classe. Cela permet par exemple à une sévérité élevée **et** incertaine d'avoir des effets différenciés sur chaque classe de risque.

Le cas scalaire $\beta_{ij} \in \mathbb{R}$ est le cas particulier $d_i = d_j = 1$.

---

## Quatre forces en tension

| Terme | Rôle | Effet |
|---|---|---|
| Couplages $\beta_{ij}$ | cohérence entre voisins | aligne les $\mathbb{E}[T]$ des nœuds liés |
| Prior relaxé $\theta_i^{\text{relax}}$ | ancrage épidémiologique + historique | ramène vers $\theta_i^{\text{epidemio}}$ via $\theta_i^{\text{prev}}$ |
| Observations $\mathcal{O}_i$ | fidélité aux données | met à jour selon les mesures disponibles |
| Entropie $\mathbb{H}$ | anti-collapse | maintient l'incertitude, évite les distributions dégénérées |

---

## Dynamique temporelle

Le prior utilisé à chaque instant $t$ n'est pas l'épidémio brute, mais l'état patient précédent **relaxé** vers l'épidémio :

$$\theta_i^{\text{relax}}(t) = (1 - e^{-\Delta t / \tau_i}) \cdot \theta_i^{\text{epidemio}} + e^{-\Delta t / \tau_i} \cdot \theta_i^{\text{prev}}(t)$$

où $\theta_i^{\text{prev}}(t) = \theta_i^{\text{post}}(t-1)$ — le posterior de l'étape précédente devient le prior non relaxé de l'étape courante.

- $\tau_i$ : constante de temps propre à chaque variable (ex: court pour glycémie, long pour statut fumeur)
- L'historique patient est encodé dans $\theta_i^{\text{prev}}$ sans stocker de trajectoire explicite
- Structure de **filtre bayésien hiérarchique** à trois échelles :

| Échelle | Mécanisme | Paramètres |
|---|---|---|
| Épidémio | prior populationnel statique | $\theta_i^{\text{epidemio}}$ |
| Patient | relaxation à $\tau_i$ | $\theta_i^{\text{relax}}(t)$ |
| Observation | mise à jour instantanée | $\mathcal{O}_i$ |

---

## Calibration

**Priors épidémiologiques $\theta_i^{\text{epidemio}}$** — directement depuis la littérature épidémiologique. Simples à justifier cliniquement.

**Couplages $\beta_{ij}$** — calibration en deux étapes.

**Étape 1 — Priors locaux par paire**

Pour chaque arête, on pose des priors gaussiens sur les paramètres de couplage. Un couplage scalaire $b \sim \mathcal{N}(\mu_b, \sigma_b^2)$ introduit 2 degrés de liberté ; une matrice $B_{ij} \in \mathbb{R}^{d_i \times d_j}$ en introduit $2 \cdot d_i \cdot d_j$.

On demande à l'expert autant d'**espérances conditionnelles interprétables** que de degrés de liberté. Exemple sur grippe/température :

- *"Température moyenne chez un patient certainement grippé ?"* → $\mu_b$
- *"À quel point êtes-vous certain de cette relation ?"* → $\sigma_b$

L'expert répond cliniquement. La traduction en contraintes sur $(\mu_b, \sigma_b)$ est faite en interne. Cette étape est **purement locale** — elle ignore le reste du graphe.

**Étape 2 — Calibration globale par patients virtuels**

Les priors de l'étape 1 régularisent une optimisation MAP sur des patients types construits avec l'expert :

$$\hat{\boldsymbol{\beta}} = \arg\max_{\boldsymbol{\beta}} \left[ \sum_m \log P(\mathbf{x}^{(m)} | \boldsymbol{\beta}) + \log P(\boldsymbol{\beta}) \right]$$

Les patients types sont auditables cliniquement et capturent les interactions globales du graphe que l'étape 1 ignore.

**Couplages variables** — si $\beta_{ij}$ semble dépendre du patient, c'est le signal qu'une variable cachée $z_{ij}$ médiatise le couplage. On l'ajoute au graphe, et $\beta_{ij}$ redevient fixe.

---

## Points ouverts

- Choix de la structure du graphe (expert vs. appris)
- Gestion du MNAR (missing not at random) dans les EHR
- Validation clinique des $\theta_i^{\text{new}}$ sur des outcomes réels
- Passage à l'échelle sur des graphes denses