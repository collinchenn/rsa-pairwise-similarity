"""
Representational Similarity Analysis (RSA) utility

Usage:
    from rsa import RSA

    rsa = RSA(model)
    rsa_scores, top_list = rsa.run("data/pairs.txt", top_k=50)
"""


from typing import Dict, Tuple, Sequence, Iterable, List, Optional
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
import pandas as pd


class RSA:
    def __init__(self, model):
        self.model = model

        # TBD: add hooks parameter allowing for further customization
        # Hooks that will be included in the analysis 
        L = model.cfg.n_layers
        hooks = (
            [("resid_pre",  l) for l in range(L)] +
            [("attn_out",   l) for l in range(L)] +
            [("mlp_out",    l) for l in range(L)] +
            [("resid_post", l) for l in range(L)]
        )
        self.hooks = list(hooks)

    # --------------------------- Public Interface ---------------------------

    def run(self, file_name: str):
        """
        Full RSA pipeline for a single human similarity file.

        Pipeline:
        1. Read human pair similarities from file
        2. Get unique items (words) and cache activations for each word
        3. Build a vector for each (word, hook)
        4. Compute cosine similarities per pair per hook
        5. Compute RSA (Pearson r) between machine and human similarities

        Args:
            file_name: Path to a text file with lines like:
                       word1#id1   word2#id2   similarity_score
                       or simply,
                       word1       word2       similarity_score

        Returns:
            rsa_scores: dict {hook_tag -> r}
            rsa_pvalues: dict {hook_tag -> p}
        """
        # 1. Read human similarities
        pair_similarities = self._read_pair_similarities(file_name)
        ordered_pairs = sorted(pair_similarities.keys())
        human_vec = np.array(
            [pair_similarities[pair] for pair in ordered_pairs], dtype=float
        )

        # 2. Get all unique items
        items = sorted({w for pair in ordered_pairs for w in pair})

        # 3. Cache activations for each word
        activation_maps = self._generate_activations(items)

        # 4. Build per-hook vectors for each word
        word_to_hook_vecs = self._build_word_to_hook_vecs(activation_maps)

        # 5. Compute cosine similarities per hook
        cosine_by_hook = self._generate_cosine_sim(
            pairs=ordered_pairs,
            word_to_hook_vecs=word_to_hook_vecs,
        )

        # 6. Compute RSA scores
        rsa_scores, rsa_pvalues = self._compute_rsa(
            cosine_by_hook=cosine_by_hook,
            human_vec=human_vec,
        )

        return rsa_scores, rsa_pvalues

    # -------------------------- Private interface --------------------------

    def _read_pair_similarities(self, file_name: str) -> Dict[Tuple[str, str], float]:
        """
        Read pair similarity values from file_name, which should be local,
        and return a mapping (word1, word2) -> similarity_score

        File format: each line like
            word1#meta    word2#meta    score
        such that each score is in [1,10]. Will be normalized to [0,1]
        """
        pair_similarities = {}

        with open(file_name, 'r') as f:
            for line in f:
                pair_1, pair_2, similiarity_score = line.split()
                pair_similarities[(pair_1.split('#', 1)[0], 
                                   pair_2.split('#', 1)[0])] = float(similiarity_score)

        # Normalize the pairs
        return {k: round(v/10, 3) for k, v in pair_similarities.items()}

    def _generate_activations(self, items: list[str]) -> Dict:
        """
        Returns activation cache for all words in `items`

        Each entry is:
            activation_maps[word] = cache
        where `cache` is the model's activation cache
        """
        activation_maps = {}

        for word in items:
            toks = self.model.to_tokens(" " + word, prepend_bos=False)
            _, cache = self.model.run_with_cache(toks, remove_batch_dim=True)
            activation_maps[word] = cache

        return activation_maps

    def _build_word_to_hook_vecs(self,
        activation_maps: Dict,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Construct mapping:
            word_to_hook_vecs[word][hook_tag] = vector (numpy array)

        hook_tag format: "{name}@{layer}"
            ex: resid_pre@1 -> Residual pre at layer 1
        """
        word_to_hook_vecs: Dict[str, Dict[str, np.ndarray]] = {}

        for word, cache in activation_maps.items():
            activations_for_word: Dict[str, np.ndarray] = {}
            for name, layer in self.hooks:
                hook_tag = f"{name}@{layer}"
                activations_for_word[hook_tag] = self._get_vector_from_cache(
                    cache, name, layer
                )
            word_to_hook_vecs[word] = activations_for_word

        return word_to_hook_vecs

    def _get_vector_from_cache(self, cache, name: str, layer: int) -> np.ndarray:
        """
        Grabs the activation matrix for a chosen hook and layer
        from the word's cache, returns as NumPy vector suitable
        for cosine similarity analysis.

        Uses last-token pooling to get one representative vetor
        """
        vector = cache[name, layer][-1]  # last-token pooling
        return vector.detach().float().cpu().numpy()

    @staticmethod
    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        """
        Returns the cosine similarity between u and v
        """
        nu, nv = norm(u), norm(v)
        if nu == 0 or nv == 0:
            return np.nan
        return float(np.dot(u, v) / (nu * nv))

    def _generate_cosine_sim(
        self,
        pairs: Sequence[Tuple[str, str]],
        word_to_hook_vecs: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """
        Compute cosine similarity for each pair of words, for each hook

        Returns:
            cosine_by_hook: {hook_tag -> np.array of cosine sims
                             aligned with `pairs` order}
        """
        cosine_by_hook: Dict[str, np.ndarray] = {}

        for name, layer in self.hooks:
            hook_tag = f"{name}@{layer}"
            machine_sim: List[float] = []

            for (w1, w2) in pairs:
                v1 = word_to_hook_vecs[w1][hook_tag]
                v2 = word_to_hook_vecs[w2][hook_tag]
                machine_sim.append(self._cosine(v1, v2))

            cosine_by_hook[hook_tag] = np.asarray(machine_sim, dtype=float)

        return cosine_by_hook

    def _compute_rsa(
        self,
        cosine_by_hook: Dict[str, np.ndarray],
        human_vec: np.ndarray,
    ) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """
        Compute RSA scores (Pearson r) for each hook

        Args:
            cosine_by_hook: {hook_tag -> machine similarity vector}
            human_vec: numpy array of human similarities (aligned with pairs)

        Returns:
            rsa_scores: full dict {hook_tag -> r}
            rsa_pvalues: full dict {hook_tag -> p}
        """
        rsa_scores: Dict[str, float] = {}
        rsa_pvalues: Dict[str, float] = {}

        for hook, machine_vec in cosine_by_hook.items():
            r, p = pearsonr(machine_vec, human_vec)
            rsa_scores[hook] = r
            rsa_pvalues[hook] = p

        return (rsa_scores, rsa_pvalues)