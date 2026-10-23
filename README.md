# cuda

# Kangaroo secp256k1 — version GPU (CUDA)

Algorithme de **Pollard (kangourous)** pour retrouver une clé privée `k` à partir de sa clé publique `Q = k·G` sur secp256k1, **quand `k` est dans un intervalle connu `[1, 2^B]`**.

- **Phase 1 – Tame** : on précalcule des « points distingués » (DP) et on les enregistre dans un fichier `tame_NNN.dp`.
- **Phase 2 – Wild** : on lance des kangourous depuis la clé publique cible jusqu'à la collision avec un DP du fichier Tame.
- Le calcul tourne sur **GPU NVIDIA** (Colab, Kaggle), avec **multi-GPU** automatique.
- Les fichiers `.dp` sont **compatibles CPU ↔ GPU**.

> **Usage prévu** : défis à intervalle connu (puzzles), recherche, récupération de vos propres clés. Ce programme ne casse pas une clé 256 bits aléatoire : le temps de calcul croît en `√(2^B)`.

---

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `kangaroo.cu` | Programme complet (CPU + GPU, un seul fichier) |
| `README.md` | Ce guide |

---

## Démarrage rapide

### Google Colab

1. Ouvrez [colab.research.google.com](https://colab.research.google.com) → **Nouveau notebook**.
2. Menu **Exécution → Modifier le type d'exécution → T4 GPU → Enregistrer**.
   (Ne choisissez pas *TPU* : le programme ne fonctionne pas dessus. Les GPU H100/A100/L4 demandent Colab Pro+.)
3. Collez et lancez les cellules de la section **« Cellules à copier-coller »** ci-dessous, dans l'ordre.

### Kaggle

1. Compte Kaggle **vérifié par téléphone** (obligatoire pour Internet et GPU).
2. [kaggle.com](https://www.kaggle.com) → **Create → New Notebook**.
3. Panneau de droite **Settings** :
   - **Accelerator** : `GPU T4 x2` (deux GPU, utilisés ensemble automatiquement) ;
   - **Internet** : `On`.
4. Collez et lancez les mêmes cellules.

---

## Cellules à copier-coller

Chaque bloc = **une cellule**. Les cellules marquées *(Python)* sont du Python normal, les autres commencent par `%%bash` ou `!`.

### Cellule 1 — Vérifier le GPU
```
!nvidia-smi -L
```
Vous devez voir au moins une ligne `GPU 0: ...`. Sinon, le type d'exécution n'est pas sur GPU.

### Cellule 2 — Dossier de travail *(Python)*
Colab : Google Drive (les fichiers survivent à la déconnexion). Kaggle : `/kaggle/working`.
```python
import os
if os.path.exists('/kaggle'):
    WORK = '/kaggle/working'
else:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/kangaroo'
os.makedirs(WORK, exist_ok=True)
%cd $WORK
```

### Cellule 3 — Télécharger et compiler
À relancer à **chaque nouvelle session** (le binaire est dans `/tmp`).
```
%%bash
rm -rf /tmp/src && git clone -q https://github.com/lawahakassou-tech/cuda.git /tmp/src
cd /tmp/src && wc -l kangaroo.cu
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
NVCC=$(which nvcc || echo /usr/local/cuda/bin/nvcc)
echo "architecture : sm_$cc"
$NVCC -O3 -std=c++17 -arch=sm_$cc -Xcompiler -pthread kangaroo.cu -o /tmp/kangaroo 2>&1 | grep -i error || echo "compilation OK"
```
Durée : 1 à 2 minutes. Un avertissement `u128_hex declared but never referenced` est sans importance.

### Cellule 4 — Autotests
```
!/tmp/kangaroo --selftest | tail -1
!/tmp/kangaroo --gpucheck
```
Résultat attendu : `Tous les tests sont OK` puis `Test GPU : tout est OK` (une ligne par GPU et par mode).
**Si un test affiche `ECHEC`, n'allez pas plus loin** et ouvrez une *issue* avec la sortie.

### Cellule 5 — Mesure de vitesse (40 s)
```
%%bash
rm -f tame_*.dp
timeout 40 /tmp/kangaroo -m 1 -r 70 -d 16 -n 0 --gpu -y --save 1000 2>&1 | tr '\r' '\n' | grep -E "^\[GPU [0-9]\]|vitesse" | tail -4
rm -f tame_*.dp
```
Affiche le nombre de GPU détectés et la vitesse en sauts/s (`M/s` = millions).

### Cellule 6 — Phase 1 : générer le fichier Tame
Exemple pour une clé dans `[1, 2^80]`. Adaptez `-r` (voir [Choisir les paramètres](#choisir-les-paramètres)).
```
!/tmp/kangaroo -m 1 -r 80 -d 20 -n 0 --gpu -y --save 120
```
- `-n 0` = tourne jusqu'à ce que vous arrêtiez la cellule (bouton ■). Le programme sauvegarde alors `tame_001.dp`.
- `--save 120` = sauvegarde automatique toutes les 120 s (protège contre une coupure de session).

Pour **reprendre / compléter** un fichier existant :
```
!/tmp/kangaroo -m 2 -f 1 -n 0 --gpu -y --save 120
```
(`-f 1` = fichier `tame_001.dp`.)

### Cellule 7 — Phase 2 : retrouver une clé privée
Remplacez la valeur de `PUB` par la clé publique (hex, compressée `02…`/`03…` ou non compressée `04…`).
```python
PUB = "02..."   # <- votre clé publique
```
```
!/tmp/kangaroo -m 3 -f 1 -k {PUB} --gpu -y
```
Le résultat (hex, décimal, WIF compressé, avec vérification `k·G == Q`) est affiché et ajouté à **`solution.txt`**. Les clés non trouvées vont dans `non_trouvees.txt`.

Plusieurs clés à la fois : mettez une clé par ligne dans un fichier `cles.txt` (dans le dossier de travail) et utilisez `-k cles.txt`.

### Cellule 8 — Test complet de bout en bout (≈ 1 minute)
Vérifie tout le circuit (Tame → Wild) sur un petit intervalle, avec des clés de test internes.
```
%%bash
rm -f tame_*.dp
/tmp/kangaroo -m 1 -r 50 -d 8 -n 500000 --gpu -y 2>&1 | tr '\r' '\n' | tail -2
/tmp/kangaroo -m 3 -f 1 --bench 3 -w 20 --gpu -y 2>&1 | tr '\r' '\n' | tail -2
rm -f tame_*.dp
```
Attendu : `BENCH : 3/3 resolues`.

---

## Tout-en-un (copier-coller direct, une seule cellule)

Pour un premier essai : installe, compile, teste et mesure la vitesse. Marche sur Colab **et** Kaggle (dossier de travail temporaire).
```
%%bash
set -e
rm -rf /tmp/src && git clone -q https://github.com/lawahakassou-tech/cuda.git /tmp/src
cd /tmp/src && wc -l kangaroo.cu
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
NVCC=$(which nvcc || echo /usr/local/cuda/bin/nvcc)
$NVCC -O3 -std=c++17 -arch=sm_$cc -Xcompiler -pthread kangaroo.cu -o /tmp/kangaroo 2>&1 | grep -i error || echo "compilation OK (sm_$cc)"
cd /tmp
/tmp/kangaroo --gpucheck
rm -f tame_*.dp
timeout 40 /tmp/kangaroo -m 1 -r 70 -d 16 -n 0 --gpu -y --save 1000 2>&1 | tr '\r' '\n' | grep -E "^\[GPU [0-9]\]|vitesse" | tail -4
```

---

## Choisir les paramètres

| Option | Signification |
|---|---|
| `-r B` | La clé est dans `[1, 2^B]`. Ex. `-r 80` |
| `-d D` | Bits de « point distingué » : un DP tous les `2^D` sauts. Plus grand = fichier plus petit mais DP plus rares |
| `-n N` | Nombre de DP à générer (`0` = jusqu'à l'arrêt) |

Au démarrage du mode 1, le programme affiche une **estimation** : opérations Tame prévues, taille du fichier `.dp`, RAM nécessaire, effort Wild moyen par clé. Lisez-la avant de laisser tourner longtemps.

Repères :
- Le coût total est de l'ordre de `√(2^B)` sauts : chaque `+2` sur `B` multiplie l'effort par 2.
- Choisissez `-d` pour que le fichier tienne dans la RAM (≈ 32 octets par DP) et dans votre espace disque.
- Si le message `tampon DP sature` apparaît, **augmentez `-d`**.
- La clé doit être dans `[1, 2^B]`. Pour un intervalle `[2^(B-1), 2^B]`, prenez simplement `-r B`.

---

## Toutes les options

| Option | Rôle | Défaut |
|---|---|---|
| `-m 1\|2\|3` | 1 = générer Tame, 2 = compléter un Tame, 3 = résoudre une clé (Wild) | demandé |
| `-r B` | Intervalle `[1, 2^B]` | auto |
| `-d D` | Bits DP | auto |
| `-n N` | Nombre de DP (mode 1/2) | auto |
| `-f F` | Fichier Tame (numéro `1` ou nom) | — |
| `-k K` | Clé publique hex **ou** fichier de clés | — |
| `-w X` | Abandon d'une clé après `X × √N` opérations Wild | auto |
| `--bench N` | Test sur N clés aléatoires internes (mode 3) | — |
| `--pub HEX` | Affiche la clé publique d'une clé privée (hex) | — |
| `--selftest` | Autotest de l'arithmétique | — |
| `-y` | Aucune question interactive (**à utiliser sur Colab/Kaggle**) | auto si non interactif |
| `--save S` | Sauvegarde auto toutes les `S` secondes | 600 |

**Options GPU**

| Option | Rôle | Défaut |
|---|---|---|
| `--gpu` | Active le GPU (**tous** les GPU visibles) | off |
| `--gpus N\|all` | Nombre de GPU à utiliser | all |
| `--gdev N` | Utilise seulement le GPU numéro N | — |
| `--gk K` | Kangourous par thread GPU | 64 |
| `--gblocks N` | Nombre de blocs (0 = automatique : 4 × nombre de SM) | auto |
| `--gthreads N` | Threads par bloc | 128 |
| `--gsteps N` | Itérations par lancement GPU | 32 |
| `--gpucheck` | Autotest des GPU | — |
| `-t N` | Ajoute N threads CPU en plus du GPU | 0 avec `--gpu` |

Les valeurs par défaut sont adaptées au T4. Mesure (T4 Colab, `-r 70 -d 16`) :

| `--gk` | Vitesse |
|---|---|
| 32 | ≈ 322 M sauts/s |
| **64** | **≈ 373 M sauts/s** |
| 128 | ≈ 374 M sauts/s |

Le multi-GPU (Kaggle `T4 x2`) est prévu pour approcher le double, mais **n'a pas encore été mesuré sur une vraie machine à deux GPU**.

---

## Fichiers générés

| Fichier | Contenu |
|---|---|
| `tame_001.dp`, `tame_002.dp`, … | Points distingués Tame (à garder !) |
| `solution.txt` | Clés privées trouvées (**ne le partagez jamais**) |
| `non_trouvees.txt` | Clés abandonnées (limite `-w` atteinte) |

### Où ils sont enregistrés
- **Colab** : `Mon Drive/kangaroo/` (persistant, tant que Drive est monté).
- **Kaggle** : `/kaggle/working/` — **temporaire**. Avant la fin de la session, téléchargez les fichiers (panneau *Output* → *Download*) ou faites *Save Version → Save & Run All (Commit)* pour les conserver.

---

## Limites des sessions

- **Colab gratuit** : session coupée après quelques heures (ou après inactivité) ; GPU non garanti. Gardez `--save 120`.
- **Kaggle** : session limitée (environ 9 à 12 h) et quota GPU hebdomadaire. Pensez à récupérer vos fichiers.
- Après coupure : relancez les cellules 2 et 3 (le binaire est recompilé), puis reprenez avec le **mode 2** (`-m 2 -f 1`).

---

## Dépannage

| Problème | Solution |
|---|---|
| `Aucun GPU CUDA detecte` | Activez un accélérateur GPU (Colab : *T4 GPU* ; Kaggle : *GPU T4 x2*) puis relancez depuis la cellule 2 |
| `nvcc: command not found` | La cellule 3 essaie déjà `/usr/local/cuda/bin/nvcc`. Sinon, signalez la sortie de `!ls /usr/local | grep cuda` |
| `git clone` échoue sur Kaggle | Activez **Internet : On** dans Settings (compte vérifié par téléphone) |
| `wc -l` n'affiche pas ~1478 | Le fichier sur GitHub est incomplet : recollez `kangaroo.cu` en entier |
| Erreur à la compilation sur `__int128` | CUDA trop ancien : il faut CUDA ≥ 11.5 (Colab et Kaggle récents conviennent) |
| `tampon DP sature` | Augmentez `-d` |
| `ECHEC` à `--gpucheck` | Ne continuez pas ; ouvrez une issue avec la sortie complète |
| Mémoire GPU insuffisante | Réduisez `--gk` (ex. 32) ou `--gblocks` |
| Très lent / 0 M/s au début | Normal pendant quelques secondes : le CPU génère les points de départ (`generation de N points de base`) |
| Arrêt de la cellule sans sauvegarde | Utilisez toujours `--save 120` |

---

## Compilation hors Colab/Kaggle

```bash
# GPU NVIDIA (remplacez sm_75 par votre architecture : T4=75, A100=80, L4=89, H100=90)
nvcc -O3 -std=c++17 -arch=sm_75 -Xcompiler -pthread kangaroo.cu -o kangaroo

# CPU seul (GCC/Clang 64 bits, sans GPU) : l'option --gpu fonctionne alors en émulation lente
g++ -O3 -march=native -std=c++17 -pthread -x c++ kangaroo.cu -o kangaroo_cpu
```

---

## Principe (résumé)

- Les kangourous avancent par sauts pseudo-aléatoires `P ← P + J[x mod NJ]` en suivant la distance parcourue.
- La symétrie `±` divise l'espace de recherche (on ne garde que les points à `y` pair).
- Sur GPU, **un thread gère `K` kangourous** et partage **une seule inversion modulaire** (astuce de Montgomery) : coût ≈ 6 multiplications par saut au lieu d'environ 270.
- Les DP trouvés par le GPU sont renvoyés au CPU, qui les enregistre (Tame) ou cherche la collision (Wild).
