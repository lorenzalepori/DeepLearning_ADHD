"""
LOOCV PARZIALE (sottoinsieme bilanciato di fold), con protezioni numeriche
contro l'esplosione dei gradienti/NaN:
 - learning rate esplicito piu' basso (1e-4 invece del default Adam 1e-3)
 - clipnorm=1.0 per tagliare i gradienti troppo grandi
 - epsilon piu' grande nella normalizzazione (1e-4 invece di 1e-8)
 - controllo esplicito di NaN nei pesi dopo ogni training, con log a schermo
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from tf_keras import layers, initializers, regularizers, Model, callbacks, optimizers
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ----------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------
GLOBAL_SEED = 42

def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_all_seeds(GLOBAL_SEED)

DATA_DIR = Path(r"C:\Users\ASUS\OneDrive\Desktop\DSAI\DeepLearning\Functions\Data\processed")

N_VAL = 10
AUG_FRACTION = 0.2
CONTRASTIVE_MARGIN = 1.0
DECISION_THRESHOLD = CONTRASTIVE_MARGIN / 2

N_FOLDS_SUBSET_PER_CLASS = 10   # 10 ADHD + 10 Control = 20 training totali

EPOCHS = 30
STEPS_PER_EPOCH = 50
PATIENCE = 6

LEARNING_RATE = 1e-4      # <-- piu' basso del default Adam (1e-3)
CLIPNORM = 1.0            # <-- taglia i gradienti troppo grandi
NORM_EPSILON = 1e-4       # <-- piu' protettivo di 1e-8 contro std vicino a zero

MODEL_INPUT_ORDER = ["alpha", "beta", "delta", "theta", "gamma"]
BAND_DISPLAY_NAMES = {"alpha": "Alpha", "beta": "Beta", "delta": "Delta",
                       "theta": "Theta", "gamma": "Gamma"}


# ----------------------------------------------------------------------
# 1. Dati
# ----------------------------------------------------------------------
def load_precomputed():
    brain_maps = np.load(DATA_DIR / "brain_maps.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    subject_ids = np.load(DATA_DIR / "subject_ids.npy")
    return brain_maps, labels, subject_ids


def split_frequency_bands(brain_maps):
    return {
        "delta": brain_maps[..., 0:4],
        "theta": brain_maps[..., 4:8],
        "alpha": brain_maps[..., 8:12],
        "beta":  brain_maps[..., 12:35],
        "gamma": brain_maps[..., 35:40],
    }


# ----------------------------------------------------------------------
# 2. Loss e distanza
# ----------------------------------------------------------------------
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, 1e-7))


def contrastive_loss(margin=CONTRASTIVE_MARGIN):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss


# ----------------------------------------------------------------------
# 3. Architettura
# ----------------------------------------------------------------------
def base_network_att(delta_shape, theta_shape, alpha_shape, beta_shape, gamma_shape,
                      embed_dim=16, num_heads=2, dropout_rate=0.1, l2_reg=1e-5, lc_init_std=1.0):

    input_alpha = layers.Input(shape=alpha_shape, name="alpha_input")
    input_beta  = layers.Input(shape=beta_shape,  name="beta_input")
    input_delta = layers.Input(shape=delta_shape, name="delta_input")
    input_theta = layers.Input(shape=theta_shape, name="theta_input")
    input_gamma = layers.Input(shape=gamma_shape, name="gamma_input")

    lc_alpha = layers.LocallyConnected2D(1, kernel_size=5, activation="hard_sigmoid",
                                          kernel_initializer=initializers.RandomNormal(stddev=lc_init_std, seed=GLOBAL_SEED+0),
                                          kernel_regularizer=regularizers.l2(l2_reg))(input_alpha)
    lc_beta = layers.LocallyConnected2D(1, kernel_size=5, activation="hard_sigmoid",
                                          kernel_initializer=initializers.RandomNormal(stddev=lc_init_std, seed=GLOBAL_SEED+1),
                                          kernel_regularizer=regularizers.l2(l2_reg))(input_beta)
    lc_delta = layers.LocallyConnected2D(1, kernel_size=5, activation="hard_sigmoid",
                                          kernel_initializer=initializers.RandomNormal(stddev=lc_init_std, seed=GLOBAL_SEED+2),
                                          kernel_regularizer=regularizers.l2(l2_reg))(input_delta)
    lc_theta = layers.LocallyConnected2D(1, kernel_size=5, activation="hard_sigmoid",
                                          kernel_initializer=initializers.RandomNormal(stddev=lc_init_std, seed=GLOBAL_SEED+3),
                                          kernel_regularizer=regularizers.l2(l2_reg))(input_theta)
    lc_gamma = layers.LocallyConnected2D(1, kernel_size=5, activation="hard_sigmoid",
                                          kernel_initializer=initializers.RandomNormal(stddev=lc_init_std, seed=GLOBAL_SEED+4),
                                          kernel_regularizer=regularizers.l2(l2_reg))(input_gamma)

    band_maps = [lc_alpha, lc_beta, lc_delta, lc_theta, lc_gamma]

    tokens = []
    for band_map in band_maps:
        pooled = layers.GlobalAveragePooling2D()(band_map)
        token = layers.Dense(embed_dim, activation="tanh",
                              kernel_regularizer=regularizers.l2(l2_reg))(pooled)
        token = layers.Dropout(dropout_rate)(token)
        tokens.append(token)

    token_seq = layers.Lambda(lambda t: tf.stack(t, axis=1))(tokens)

    attn_output, attn_scores = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim, name="band_attention"
    )(token_seq, token_seq, return_attention_scores=True)

    x = layers.Add()([token_seq, attn_output])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(embed_dim, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg))(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    importance_score = layers.Dense(1, activation="sigmoid")(x)
    importance_score = layers.Reshape((5,), name="importance_score")(importance_score)

    weighted_maps = []
    for i, band_map in enumerate(band_maps):
        weight = layers.Lambda(lambda t, idx=i: t[:, idx:idx+1])(importance_score)
        weight = layers.Reshape((1, 1, 1))(weight)
        weighted_maps.append(layers.Multiply()([band_map, weight]))

    merged = layers.Concatenate(axis=-1)(weighted_maps)

    y = layers.Conv2D(16, kernel_size=5, activation="tanh",
                       kernel_regularizer=regularizers.l2(l2_reg))(merged)
    y = layers.Conv2D(16, kernel_size=5, activation="tanh",
                       kernel_regularizer=regularizers.l2(l2_reg))(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(dropout_rate)(y)
    embedding = layers.Dense(16, activation="tanh", name="embedding",
                              kernel_regularizer=regularizers.l2(l2_reg))(y)

    return Model(
        inputs=[input_alpha, input_beta, input_delta, input_theta, input_gamma],
        outputs=[embedding, importance_score],
        name="base_network_att"
    )


def build_siamese(base_network, alpha_shape, beta_shape, delta_shape, theta_shape, gamma_shape):
    alpha_A = layers.Input(alpha_shape); beta_A = layers.Input(beta_shape)
    delta_A = layers.Input(delta_shape); theta_A = layers.Input(theta_shape)
    gamma_A = layers.Input(gamma_shape)

    alpha_B = layers.Input(alpha_shape); beta_B = layers.Input(beta_shape)
    delta_B = layers.Input(delta_shape); theta_B = layers.Input(theta_shape)
    gamma_B = layers.Input(gamma_shape)

    embedded_A, _ = base_network([alpha_A, beta_A, delta_A, theta_A, gamma_A])
    embedded_B, _ = base_network([alpha_B, beta_B, delta_B, theta_B, gamma_B])

    distance = layers.Lambda(euclidean_distance, name="distance")([embedded_A, embedded_B])

    model = Model(
        inputs=[[alpha_A, beta_A, delta_A, theta_A, gamma_A],
                [alpha_B, beta_B, delta_B, theta_B, gamma_B]],
        outputs=distance
    )
    return model


# ----------------------------------------------------------------------
# 4. Split LOOCV completo + selezione sottoinsieme bilanciato
# ----------------------------------------------------------------------
def data_split_loocv(labels, n_val=N_VAL, seed=GLOBAL_SEED):
    n_subjects = len(labels)
    all_idx = np.arange(n_subjects)
    rng = np.random.RandomState(seed)
    folds = []
    for test_idx_single in range(n_subjects):
        remaining = np.setdiff1d(all_idx, [test_idx_single])
        rng.shuffle(remaining)
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]
        folds.append({"test_idx": np.array([test_idx_single]), "val_idx": val_idx, "train_idx": train_idx})
    return folds


def select_balanced_subset(folds, labels, n_per_class=N_FOLDS_SUBSET_PER_CLASS, seed=GLOBAL_SEED):
    rng = np.random.RandomState(seed)
    adhd_fold_indices = [i for i, f in enumerate(folds) if labels[f["test_idx"][0]] == 1]
    control_fold_indices = [i for i, f in enumerate(folds) if labels[f["test_idx"][0]] == 0]
    rng.shuffle(adhd_fold_indices)
    rng.shuffle(control_fold_indices)
    chosen = adhd_fold_indices[:n_per_class] + control_fold_indices[:n_per_class]
    rng.shuffle(chosen)
    return [folds[i] for i in chosen]


# ----------------------------------------------------------------------
# 5. Coppie e augmentation
# ----------------------------------------------------------------------
def create_pairs(indices, labels):
    pos_pairs, neg_pairs = [], []
    idx_list = list(indices)
    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            a, b = idx_list[i], idx_list[j]
            if labels[a] == labels[b]:
                pos_pairs.append((a, b))
            else:
                neg_pairs.append((a, b))
    return pos_pairs, neg_pairs


def augment_batch(batch_bands, batch_labels, all_bands, all_labels, valid_pool_idx,
                   aug_fraction=AUG_FRACTION, band_names_keys=MODEL_INPUT_ORDER):
    batch_bands = {k: v.copy() for k, v in batch_bands.items()}
    n = len(batch_labels)
    for band_name in band_names_keys:
        n_aug = max(1, int(n * aug_fraction))
        aug_positions = np.random.choice(n, size=n_aug, replace=False)
        for pos in aug_positions:
            same_class_pool = valid_pool_idx[all_labels[valid_pool_idx] == batch_labels[pos]]
            if len(same_class_pool) == 0:
                continue
            donor = np.random.choice(same_class_pool)
            batch_bands[band_name][pos] = all_bands[band_name][donor]
    return batch_bands


def data_generator(pos_pairs, neg_pairs, bands_norm, labels, train_idx, batch_size=16,
                    augment=True, band_names_keys=MODEL_INPUT_ORDER):
    pos_pairs = list(pos_pairs)
    neg_pairs = list(neg_pairs)
    half = batch_size // 2
    while True:
        np.random.shuffle(pos_pairs)
        np.random.shuffle(neg_pairs)
        chosen_pos = [pos_pairs[i % len(pos_pairs)] for i in range(half)]
        chosen_neg = [neg_pairs[i % len(neg_pairs)] for i in range(half)]
        all_pairs = chosen_pos + chosen_neg
        y = np.array([1] * len(chosen_pos) + [0] * len(chosen_neg))
        idx_A = np.array([p[0] for p in all_pairs])
        idx_B = np.array([p[1] for p in all_pairs])
        batch_A = {k: bands_norm[k][idx_A] for k in band_names_keys}
        batch_B = {k: bands_norm[k][idx_B] for k in band_names_keys}
        if augment:
            batch_A = augment_batch(batch_A, labels[idx_A], bands_norm, labels, train_idx, band_names_keys=band_names_keys)
            batch_B = augment_batch(batch_B, labels[idx_B], bands_norm, labels, train_idx, band_names_keys=band_names_keys)
        x1 = [batch_A[k] for k in band_names_keys]
        x2 = [batch_B[k] for k in band_names_keys]
        yield (x1, x2), y


# ----------------------------------------------------------------------
# 6. Train di un fold, con protezioni numeriche
# ----------------------------------------------------------------------
def train_fold(fold, bands_raw, labels, epochs=EPOCHS,
                steps_per_epoch=STEPS_PER_EPOCH, patience=PATIENCE, verbose=0):

    train_idx = fold["train_idx"]
    val_idx = fold["val_idx"]

    norm_stats = {}
    bands_norm = {}
    for band_name, arr in bands_raw.items():
        mean = arr[train_idx].mean()
        std = arr[train_idx].std() + NORM_EPSILON   
        norm_stats[band_name] = (mean, std)
        bands_norm[band_name] = (arr - mean) / std

    shapes = {k: bands_norm[k].shape[1:] for k in MODEL_INPUT_ORDER}

    base_network = base_network_att(
        delta_shape=shapes["delta"], theta_shape=shapes["theta"], alpha_shape=shapes["alpha"],
        beta_shape=shapes["beta"], gamma_shape=shapes["gamma"]
    )
    siamese_model = build_siamese(
        base_network,
        alpha_shape=shapes["alpha"], beta_shape=shapes["beta"], delta_shape=shapes["delta"],
        theta_shape=shapes["theta"], gamma_shape=shapes["gamma"]
    )

    opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)
    siamese_model.compile(optimizer=opt, loss=contrastive_loss(CONTRASTIVE_MARGIN))

    pos_train, neg_train = create_pairs(train_idx, labels)
    pos_val, neg_val = create_pairs(val_idx, labels)

    train_gen = data_generator(pos_train, neg_train, bands_norm, labels, train_idx, batch_size=16, augment=True)
    val_gen = data_generator(pos_val, neg_val, bands_norm, labels, val_idx, batch_size=16, augment=False)

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = siamese_model.fit(
        train_gen, steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, validation_steps=5,
        epochs=epochs, callbacks=[early_stop], verbose=verbose
    )

    weights = base_network.get_weights()
    has_nan = any(np.isnan(w).any() for w in weights)
    if has_nan:
        print("  [ATTENZIONE] I pesi di questo fold contengono NaN - risultati inaffidabili per questo fold.")

    return siamese_model, norm_stats, bands_norm, base_network, has_nan


def evaluate_fold_majority_vote(fold, bands_norm, labels, subject_ids, siamese_model,
                                 threshold=DECISION_THRESHOLD):
    train_idx = fold["train_idx"]
    test_idx = fold["test_idx"][0]

    adhd_train_idx = train_idx[labels[train_idx] == 1]
    n_ref = len(adhd_train_idx)
    ref_inputs = [bands_norm[name][adhd_train_idx] for name in MODEL_INPUT_ORDER]

    test_inputs = [bands_norm[name][test_idx:test_idx+1] for name in MODEL_INPUT_ORDER]
    x1 = [np.repeat(t, n_ref, axis=0) for t in test_inputs]
    x2 = ref_inputs

    distances = siamese_model.predict([x1, x2], verbose=0).ravel()
    per_pair_votes = (distances < threshold).astype(int)

    predicted_label = int(per_pair_votes.mean() > 0.5)
    true_label = int(labels[test_idx])
    score = float(per_pair_votes.mean())

    return predicted_label, true_label, score, subject_ids[test_idx]


if __name__ == "__main__":
    print("Caricamento dati...")
    brain_maps, labels, subject_ids = load_precomputed()
    bands_raw = split_frequency_bands(brain_maps)

    print("Creazione di tutti i 121 fold LOOCV...")
    all_folds = data_split_loocv(labels, n_val=N_VAL, seed=GLOBAL_SEED)

    print(f"Selezione di un sottoinsieme bilanciato: {N_FOLDS_SUBSET_PER_CLASS} fold ADHD "
          f"+ {N_FOLDS_SUBSET_PER_CLASS} fold Control")
    folds_subset = select_balanced_subset(all_folds, labels, n_per_class=N_FOLDS_SUBSET_PER_CLASS)

    y_true, y_pred, y_score, subj_out = [], [], [], []
    importance_records = []
    n_nan_folds = 0

    for i, fold in enumerate(folds_subset):
        test_subj = subject_ids[fold["test_idx"][0]]
        test_label = "ADHD" if labels[fold["test_idx"][0]] == 1 else "Control"
        print(f"\n=== Fold {i+1}/{len(folds_subset)} - test subject {test_subj} ({test_label}) ===")

        set_all_seeds(seed=GLOBAL_SEED + i)
        siamese_model, norm_stats, bands_norm, base_network, has_nan = train_fold(
            fold, bands_raw, labels, epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH, patience=PATIENCE, verbose=0
        )

        if has_nan:
            n_nan_folds += 1
            print("  Fold SALTATO (pesi NaN) - non incluso nelle metriche finali.")
            continue

        pred, true, score, subj = evaluate_fold_majority_vote(
            fold, bands_norm, labels, subject_ids, siamese_model
        )
        y_true.append(true); y_pred.append(pred); y_score.append(score); subj_out.append(subj)
        print(f"  pred={pred} true={true} vote_fraction={score:.3f} {'OK' if pred==true else 'ERRORE'}")

        test_idx = fold["test_idx"][0]
        test_inputs = [bands_norm[name][test_idx:test_idx+1] for name in MODEL_INPUT_ORDER]
        _, imp_score = base_network.predict(test_inputs, verbose=0)

        if np.isnan(imp_score).any():
            print("  [ATTENZIONE] importance_score contiene NaN per questo soggetto - escluso dalla media.")
            continue

        importance_records.append((imp_score[0], true))

    print(f"\nFold totali saltati per NaN: {n_nan_folds}/{len(folds_subset)}")

    if len(y_true) == 0:
        print("Nessun fold valido completato - impossibile calcolare le metriche.")
    else:
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
        print(f"\n--- LOOCV parziale ({len(y_true)} fold validi) ---")
        print(f"Accuracy: {acc:.3f}   AUC: {auc:.3f}")
        print("Confusion matrix:\n", cm)

    if len(importance_records) == 0:
        print("\nNessun importance_score valido raccolto - impossibile fare il confronto ADHD vs Control.")
    else:
        importance_array = np.array([r[0] for r in importance_records])
        labels_tested = np.array([r[1] for r in importance_records])

        n_adhd_valid = (labels_tested == 1).sum()
        n_control_valid = (labels_tested == 0).sum()
        print(f"\nSoggetti ADHD validi: {n_adhd_valid}   Soggetti Control validi: {n_control_valid}")

        if n_adhd_valid == 0 or n_control_valid == 0:
            print("Manca almeno una classe tra i risultati validi - impossibile confrontare ADHD vs Control.")
        else:
            mean_adhd = importance_array[labels_tested == 1].mean(axis=0)
            mean_control = importance_array[labels_tested == 0].mean(axis=0)

            display_order = [BAND_DISPLAY_NAMES[k] for k in MODEL_INPUT_ORDER]
            print("Importance score medio:")
            print("  ADHD:   ", dict(zip(display_order, np.round(mean_adhd, 3))))
            print("  Control:", dict(zip(display_order, np.round(mean_control, 3))))

            x = np.arange(len(display_order))
            width = 0.35
            plt.figure(figsize=(7, 4))
            plt.bar(x - width/2, mean_adhd, width, label="ADHD", color="#d1495b")
            plt.bar(x + width/2, mean_control, width, label="Control", color="#2a9d8f")
            plt.xticks(x, display_order)
            plt.ylabel("Mean importance score")
            plt.ylim(0, 1)
            plt.legend()
            plt.title(f"Importance score medio (LOOCV parziale, {len(importance_records)} soggetti validi)")
            plt.tight_layout()
            plt.savefig("importance_loocv_partial.png", dpi=150)
            plt.show()