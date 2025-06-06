import numpy as np
import tensorflow as tf


# Define the BADGE active learning strategy class
class BADGE: 
    def __init__(self, model, pool_dataset, batch_size=32, num_samples=100):
        self.model = model
        self.pool_dataset = pool_dataset
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.intermediate_model = self.build_intermediate_model()

    def find_last_dense_before_softmax(self):
        """
        Finds the index of the last Dense layer before the softmax activation in a Keras model.
        Returns:
            int: Index of the last Dense layer before softmax.
        Raises:
            ValueError: If no suitable Dense layer is found.
        """
        for i in reversed(range(len(self.model.layers))):
            layer = self.model.layers[i]
            if isinstance(layer, tf.keras.layers.Dense):
                activation = getattr(layer, 'activation', None)
                # Check if the activation is not softmax
                if activation is not None and activation.__name__ != 'softmax':
                    return i
        raise ValueError("No suitable dense layer found.")

    def build_intermediate_model(self):
        """
        Builds a Keras model that outputs the activations of the last Dense layer before softmax.
        Returns:
            tf.keras.Model: Intermediate model for feature extraction.
        """
        index = self.find_last_dense_before_softmax()
        return tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.layers[index].output
        )
    
    def extract_embeddings_and_probs(self, X_pool):
        """
        Uses intermediate and final model to get feature activations and class probabilities.
        """
        embeddings = self.intermediate_model.predict(X_pool, batch_size=self.batch_size)
        if X_pool.shape[0] == 0:
            raise ValueError("Er zijn geen afbeeldingen geladen in de pool. Controleer of het datasetpad correct is en of de afbeeldingen aanwezig zijn.")
        else:
            probs = self.model.predict(X_pool, batch_size=self.batch_size)
        return embeddings, probs


    def select(self):
        """
        Selects a subset of the pool using KMeans++ on gradient embeddings.
        Returns:
            List[int]: Selected indices from pool_dataset
        """
        X_pool = self.pool_dataset
        embs, probs = self.extract_embeddings_and_probs(X_pool)

        # Convert to gradient embeddings (approximatie zoals BADGE doet)
        pseudo_labels = np.argmax(probs, axis=1)
        probs = -probs
        probs[np.arange(len(probs)), pseudo_labels] += 1

        emb_norm_sq = np.sum(embs ** 2, axis=1)
        prob_norm_sq = np.sum(probs ** 2, axis=1)
        chosen, chosen_list, mu, D2 = set(), [], None, None

        for _ in range(self.num_samples):
            if len(chosen) == 0:
                ind = np.argmax(prob_norm_sq * emb_norm_sq)
                mu = [((probs[ind], prob_norm_sq[ind]), (embs[ind], emb_norm_sq[ind]))]
                D2 = self.distance((probs, prob_norm_sq), (embs, emb_norm_sq), mu[0])
                D2[ind] = 0
            else:
                newD = self.distance((probs, prob_norm_sq), (embs, emb_norm_sq), mu[-1])
                D2 = np.minimum(D2, newD)
                D2[list(chosen_list)] = 0
                P = (D2 ** 2) / np.sum(D2 ** 2)
                ind = np.random.choice(len(D2), p=P)
                while ind in chosen:
                    ind = np.random.choice(len(D2), p=P)
                mu.append(((probs[ind], prob_norm_sq[ind]), (embs[ind], emb_norm_sq[ind])))
            chosen.add(ind)
            chosen_list.append(ind)

        #print(f"Selected {len(chosen_list)} indices via BADGE.")

        return chosen_list


    def distance(self, X1, X2, mu):
        X1_vec, X1_norm = X1
        X2_vec, X2_norm = X2
        Y1_vec, Y1_norm = mu[0]
        Y2_vec, Y2_norm = mu[1]

        dists = (
            X1_norm * X2_norm
            + Y1_norm * Y2_norm
            - 2 * np.sum(X1_vec * Y1_vec, axis=1) * np.sum(X2_vec * Y2_vec, axis=1)
        )
        return np.sqrt(np.clip(dists, a_min=0.0, a_max=None))