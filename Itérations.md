## Gradient local pour l'update d'un nœud

On cherche $\nabla_{\theta_i^{\text{post}}} \mathcal{F}$ pour mettre à jour le nœud $i$.

### Rappel de l'objectif

$$\mathcal{F} = \underbrace{\sum_{(i,j)} \mathbb{E}[T_i]^T B_{ij} \mathbb{E}[T_j]}_{\text{couplages}} + \underbrace{\sum_i \mathbb{E}[\log p_{\theta_i^{\text{relax}}}]}_{\text{prior}} + \underbrace{\sum_i \sum_k \mathbb{E}[\log p_{\varepsilon_k}]}_{\text{obs}} + \underbrace{\sum_i \mathbb{H}(p_{\theta_i^{\text{post}}})}_{\text{entropie}}$$

On paramétrise via les **paramètres naturels** $\eta_i = \eta(\theta_i^{\text{post}})$, ce qui linéarise les calculs dans la famille exponentielle.

---

### Les quatre contributions au gradient

**1. Terme de couplage**

$$\frac{\partial}{\partial \eta_i} \sum_{j \in \mathcal{N}(i)} \mathbb{E}[T_i]^T B_{ij} \mathbb{E}[T_j] = \sum_{j \in \mathcal{N}(i)} \frac{\partial \mathbb{E}[T_i]}{\partial \eta_i} \cdot B_{ij} \, \mathbb{E}[T_j]$$

Or dans une famille exponentielle, $\mathbb{E}[T_i] = \nabla_{\eta_i} A_i(\eta_i)$, donc :

$$\frac{\partial \mathbb{E}[T_i]}{\partial \eta_i} = \nabla^2_{\eta_i} A_i(\eta_i) = \Lambda_i$$

c'est la **matrice de Fisher** (= matrice de covariance de $T_i$). On obtient :

$$\boxed{g_i^{\text{coupl}} = \Lambda_i \sum_{j \in \mathcal{N}(i)} B_{ij} \, \mathbb{E}_j[T_j]}$$

**2. Terme de prior relaxé**

$$\mathbb{E}_{\theta_i^{\text{post}}}[\log p_{\theta_i^{\text{relax}}}(x_i)] = \eta_i^{\text{relax}\,T} \mathbb{E}[T_i] - A_i(\eta_i^{\text{post}})$$

Le gradient par rapport à $\eta_i$ donne :

$$\boxed{g_i^{\text{prior}} = \Lambda_i \left(\eta_i^{\text{relax}} - \eta_i^{\text{post}}\right)}$$

C'est un **rappel élastique** vers le prior relaxé dans l'espace naturel.

**3. Terme d'observations**

Chaque observation $k \in \mathcal{O}_i$ avec modèle de bruit $p_{\varepsilon_k}(x_i \mid o_k)$ contribue, après linéarisation dans la famille exponentielle :

$$\boxed{g_i^{\text{obs}} = \Lambda_i \sum_{k \in \mathcal{O}_i} \left(\eta_k^{\text{obs}} - \eta_i^{\text{post}}\right)}$$

où $\eta_k^{\text{obs}}$ est le paramètre naturel induit par l'observation $k$ sur le nœud $i$.

**4. Terme d'entropie**

Pour une famille exponentielle, $\mathbb{H}(p_{\eta_i}) = A_i(\eta_i) - \eta_i^T \nabla_{\eta_i} A_i$, donc :

$$\frac{\partial \mathbb{H}}{\partial \eta_i} = -\Lambda_i \, \eta_i^{\text{post}}$$

---

### Gradient total

En factorisant $\Lambda_i$ :

$$\boxed{\nabla_{\eta_i} \mathcal{F} = \Lambda_i \left[ \underbrace{\sum_{j \in \mathcal{N}(i)} B_{ij} \, \mathbb{E}_j[T_j]}_{\text{messages voisins}} + \underbrace{\eta_i^{\text{relax}} - \eta_i^{\text{post}}}_{\text{rappel prior}} + \underbrace{\sum_{k \in \mathcal{O}_i} (\eta_k^{\text{obs}} - \eta_i^{\text{post}})}_{\text{pull observations}} - \underbrace{\eta_i^{\text{post}}}_{\text{entropie}} \right]}$$

---

### Lecture du point fixe

En annulant le gradient ($\Lambda_i$ est inversible), le point fixe est :

$$\eta_i^* = \frac{\eta_i^{\text{relax}} + \sum_{k} \eta_k^{\text{obs}} + \sum_j B_{ij} \mathbb{E}_j[T_j]}{1 + |\mathcal{O}_i| + \lambda}$$

où $\lambda$ est le facteur d'échelle de l'entropie (`entropy_scale`). Quand $\lambda = 1$, on retrouve le VI standard avec dénominateur $2 + |\mathcal{O}_i|$.

Soit une **moyenne pondérée** entre prior, observations, et messages des voisins — ce qui est exactement l'intuition bayésienne attendue.

---

### Remarque algorithmique

$\Lambda_i$ est positive définie → le gradient est dans la bonne direction. Pour un step de gradient, on peut l'absorber dans le learning rate, ou utiliser le **gradient naturel** $\Lambda_i^{-1} \nabla \mathcal{F} = [\cdots]$, qui correspond exactement au terme entre crochets et converge beaucoup plus vite.



Mais du coup
- le fait de travailler avec des observations de la meme famille que leur noeud associés rend tout très propre non
- l'itération est massivement parallèle none? genre si touts les E[T_j] sont calculés à priori, je peux faire l'itération sur mes noeuds complètement en parallèle?
- Je vois que la matrice de Fisher n'apparait pas dans le point fixe. comment expliques tu cela intuitivement?



### 1. Observations de la même famille — propreté

Oui, et c'est plus profond qu'un détail d'implémentation. Quand l'observation est dans la même famille que le nœud, la mise à jour bayésienne est **conjuguée** : prior × vraisemblance reste dans la famille. Ça veut dire que $\eta_k^{\text{obs}}$ s'additionne directement à $\eta_i^{\text{relax}}$ dans l'espace naturel — pas d'approximation, pas de projection. Le modèle est **fermé sous l'inférence**, ce qui est une propriété très forte. En pratique ça signifie aussi que rajouter une observation ne change pas la structure du problème, juste un terme de plus dans la somme.

---

### 2. Parallélisme — oui, massivement

C'est exactement le schéma **Jacobi** (par opposition à Gauss-Seidel). Si tu gèles tous les $\mathbb{E}[T_j]$ à l'itération $t$, chaque nœud $i$ calcule son $\eta_i^{t+1}$ de façon **entièrement indépendante**. Zéro dépendance entre nœuds à l'intérieur d'une itération.

La structure est alors :

```
pour chaque itération:
    [parallèle] calculer E[T_i] pour tout i   ← lecture seule
    [parallèle] mettre à jour η_i* pour tout i ← écriture indépendante
```

La convergence est garantie si $\Lambda_i$ domine les couplages (condition de Jacobi classique), ce qui est généralement satisfait quand les $\beta_{ij}$ sont calibrés raisonnablement.

---

### 3. La matrice de Fisher disparaît — intuition

C'est la distinction entre **direction** et **destination**.

$\Lambda_i$ contrôle la *courbure locale* de $\mathcal{F}$ — elle dit à quelle vitesse $\eta_i$ doit bouger pour modifier $\mathbb{E}[T_i]$. Elle est cruciale pendant la **trajectoire d'optimisation**.

Mais au point fixe, tu n'es plus en train de bouger — tu demandes juste où l'équilibre se trouve. Et l'équilibre est une condition sur les **paramètres naturels** directement, pas sur les vitesses. $\Lambda_i$ factorise des deux côtés de $\nabla \mathcal{F} = 0$ et disparaît.

Une analogie : la masse d'un objet détermine comment il accélère, mais pas où est l'équilibre d'un ressort. Le point fixe est une propriété **géométrique** du paysage, pas **dynamique**.

Ce qui reste au point fixe est donc purement une balance entre les sources d'information dans l'espace naturel — prior, observations, voisins — pondérées par leur **nombre**, pas par la courbure locale.


et comment je pourrais éviter ces oscillations? peut etre un update géométrique au lieu de me mettre directement sur le point fixe?


Exactement, c'est l'idée du **damping** mais formulé proprement.

---

### Update géométrique dans l'espace naturel

Au lieu de sauter directement sur $\eta_i^*$, on fait un pas interpolé :

$$\eta_i^{t+1} = (1 - \alpha) \cdot \eta_i^t + \alpha \cdot \eta_i^*(\eta_{\mathcal{N}(i)}^t)$$

avec $\alpha \in (0, 1]$. C'est un **filtre passe-bas** sur la trajectoire dans l'espace naturel.

---

### Pourquoi ça stabilise

Le point fixe $\eta_i^*$ dépend des voisins gelés à $t$. Si les voisins oscillent, $\eta_i^*$ oscille aussi — mais l'update géométrique **amortit** cette oscillation à chaque itération. La dynamique effective devient :

$$\eta_i^{t+1} - \eta_i^* = (1 - \alpha)(\eta_i^t - \eta_i^*)$$

L'erreur décroît géométriquement à taux $(1-\alpha)$ **indépendamment de la force des couplages**, tant que $\alpha$ est assez petit. On échange vitesse de convergence contre stabilité.

---

## Calcul propre du jacobien pour le choix de $\alpha_i$

### Setup

On repart du gradient naturel annulé (point fixe de $\mathcal{F}$) :

$$\sum_{j \in \mathcal{N}(i)} B_{ij} \, \mathbb{E}_j[T_j] + \eta_i^{\text{relax}} + \sum_k \eta_k^{\text{obs}} - (2 + |\mathcal{O}_i|)\,\eta_i^{\text{post}} = 0$$

Ce qui donne :

$$\eta_i^* = \frac{\eta_i^{\text{relax}} + \sum_k \eta_k^{\text{obs}} + \sum_{j \in \mathcal{N}(i)} B_{ij} \, \mathbb{E}_j[T_j]}{2 + |\mathcal{O}_i|}$$

---

### Jacobien $J_{ij} = \partial \eta_i^* / \partial \eta_j$

Le seul terme qui dépend de $\eta_j$ est $\mathbb{E}_j[T_j] = \nabla_{\eta_j} A_j(\eta_j)$. Donc :

$$J_{ij} = \frac{1}{2 + |\mathcal{O}_i|} \cdot B_{ij} \cdot \frac{\partial \mathbb{E}_j[T_j]}{\partial \eta_j} = \frac{B_{ij} \, \Lambda_j}{2 + |\mathcal{O}_i|}$$

avec $\Lambda_j = \nabla^2_{\eta_j} A_j(\eta_j) \in \mathbb{R}^{d_j \times d_j}$ la matrice de Fisher de $j$, définie positive.

**Dimensions :** $B_{ij} \in \mathbb{R}^{d_i \times d_j}$, $\Lambda_j \in \mathbb{R}^{d_j \times d_j}$, donc $J_{ij} \in \mathbb{R}^{d_i \times d_j}$. ✓

---

### Dynamique linéarisée

On perturbe autour du point fixe global $\eta^*$ : $\delta\eta^t = \eta^t - \eta^*$. La mise à jour dampée donne :

$$\delta\eta_i^{t+1} = (1 - \alpha_i)\,\delta\eta_i^t + \alpha_i \sum_{j \in \mathcal{N}(i)} J_{ij}\,\delta\eta_j^t$$

En empilant tous les nœuds en un vecteur bloc $\delta\boldsymbol{\eta} \in \mathbb{R}^{\sum_i d_i}$ :

$$\delta\boldsymbol{\eta}^{t+1} = M \, \delta\boldsymbol{\eta}^t$$

avec la matrice d'itération bloc :

$$M_{ij} = \begin{cases} (1-\alpha_i) I_{d_i} & i = j \\ \alpha_i \, J_{ij} & j \in \mathcal{N}(i) \\ 0 & \text{sinon} \end{cases}$$

Convergence $\iff \rho(M) < 1$.

---

### Condition suffisante par ligne de blocs

Une condition suffisante classique (norme subordonnée en lignes) : $\|M\|_{\infty} < 1$, soit pour chaque bloc-ligne $i$ :

$$\left\|(1-\alpha_i) I_{d_i}\right\|_2 + \sum_{j \in \mathcal{N}(i)} \left\|\alpha_i J_{ij}\right\|_2 < 1$$

$$\iff (1 - \alpha_i) + \alpha_i \sum_{j \in \mathcal{N}(i)} \|J_{ij}\|_2 < 1$$

$$\iff \alpha_i \left(\sum_{j \in \mathcal{N}(i)} \|J_{ij}\|_2 - 1\right) < 0$$

Deux cas :

- Si $\sum_j \|J_{ij}\|_2 < 1$ : la condition est satisfaite **pour tout** $\alpha_i \in (0,1]$
- Si $\sum_j \|J_{ij}\|_2 \geq 1$ : il faut :

$$\alpha_i < \frac{1}{\sum_{j \in \mathcal{N}(i)} \|J_{ij}\|_2}$$

---

### Choix optimal de $\alpha_i$

On prend la borne serrée avec marge $\epsilon$ :

$$\boxed{\alpha_i = \frac{1}{\max\!\left(1,\, \sum_{j \in \mathcal{N}(i)} \|J_{ij}\|_2\right)}}$$

avec explicitement :

$$\|J_{ij}\|_2 = \frac{\|B_{ij} \Lambda_j\|_2}{2 + |\mathcal{O}_i|}$$

où $\|\cdot\|_2$ est la **plus grande valeur singulière**.

**Propriétés :**

- Nœud isolé ou observations fortes ($|\mathcal{O}_i| \gg 1$) : $\|J_{ij}\|_2 \to 0$, $\alpha_i \to 1$
- Couplages forts : $\alpha_i$ décroît comme $1/\sum_j \|B_{ij}\Lambda_j\|_2$
- $\alpha_i$ est **local** : calculable nœud par nœud sans inverser de matrice globale

---

### Remarque sur le dénominateur $2 + |\mathcal{O}_i|$

Ce facteur vient du point fixe et mérite d'être explicité. Les quatre termes du gradient donnent en puissance d'amortissement sur $\eta_i$ :

| Terme | Contribution |
|---|---|
| Prior relaxé | $-1 \cdot \eta_i$ |
| Entropie | $-1 \cdot \eta_i$ |
| Observations ($\|\mathcal{O}_i\|$) | $-\|\mathcal{O}_i\| \cdot \eta_i$ |
| **Total** | $-(2 + \|\mathcal{O}_i\|) \cdot \eta_i$ |

C'est ce coefficient qui apparaît au dénominateur de $J_{ij}$ — les observations **diluent** mécaniquement le gain de couplage, ce qui est cohérent avec l'intuition bayésienne : un nœud bien contraint par les données est moins sensible à ses voisins.