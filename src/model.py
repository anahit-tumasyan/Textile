"""
model.py
========

Quantitative blend-composition regressors. The task: predict cotton mass
fraction (polyester = 1 - cotton) from a preprocessed NIR spectrum.

The model zoo, and why each one is here:

  * PLSRegression  -> the chemometrics standard; this is the SOTA-style BASELINE
                      the brief's Phase 0 requires us to reproduce and beat.
  * RandomForest   -> a non-linear ML comparator.
  * PCR            -> PCA + linear regression. The WP2 task document asks for
                      "PCA"; PCA alone is a decomposition, not a predictor, so
                      the standard way to benchmark it on a REGRESSION task is
                      principal component regression.
  * SVR            -> the WP2 "SVM", in its regression form. An SVM classifier
                      cannot output a continuous percentage.
  * MLP            -> plain neural network, no spatial structure assumed.
  * CNN            -> 1-D convolutional network over the wavelength axis. This is
                      the only model here that uses the fact that neighbouring
                      wavelengths are related. Requires torch; skipped with a
                      clear message if torch is absent.

NOTE ON THE TASK DOCUMENT: WP2 asks for accuracy / precision / recall / F1.
Those are classification metrics and they do not apply to this task, which
predicts a continuous percentage. Every model here is therefore scored with
MAE / RMSE / R2, the WP3 metrics, so that "improvement over baseline" compares
like with like. See docs/ for the fuller note.

Predictions are clipped to [0, 1] so they are physically valid fractions.
"""

from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#: Model keys the benchmark can run, in the order WP2 lists them.
KINDS = ("pls", "pcr", "svr", "rf", "mlp", "cnn")


class BlendRegressor:
    """Thin wrapper that clips predictions to a valid fraction range."""

    def __init__(self, kind: str = "pls", **kwargs):
        self.kind = kind
        if kind == "pls":
            self.model = PLSRegression(n_components=kwargs.get("n_components", 12))
        elif kind == "pcr":
            # Scale first: PCA on unscaled spectra is dominated by the few
            # wavelengths with the largest raw absorbance, not the informative ones.
            self.model = make_pipeline(
                StandardScaler(),
                PCA(n_components=kwargs.get("n_components", 20),
                    random_state=kwargs.get("random_state", 0)),
                LinearRegression(),
            )
        elif kind == "svr":
            self.model = make_pipeline(
                StandardScaler(),
                SVR(kernel=kwargs.get("kernel", "rbf"),
                    C=kwargs.get("C", 10.0),
                    epsilon=kwargs.get("epsilon", 0.01),
                    gamma=kwargs.get("gamma", "scale")),
            )
        elif kind == "mlp":
            self.model = make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=kwargs.get("hidden", (256, 64)),
                             alpha=kwargs.get("alpha", 1e-3),
                             learning_rate_init=1e-3,
                             max_iter=kwargs.get("max_iter", 600),
                             early_stopping=True, n_iter_no_change=25,
                             random_state=kwargs.get("random_state", 0)),
            )
        elif kind == "cnn":
            self.model = _SpectralCNN(**kwargs)
        elif kind == "rf":
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                random_state=kwargs.get("random_state", 0),
                n_jobs=-1,
            )
        else:
            raise ValueError(kind)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        pred = np.asarray(self.model.predict(X)).ravel()
        return np.clip(pred, 0.0, 1.0)


class _SpectralCNN:
    """
    Small 1-D CNN over the wavelength axis, with a scikit-learn-shaped API.

    Deliberately small (~40k parameters). A spectrum is 601 points and the
    training set is ~1200 samples; anything larger memorises. The convolutions
    are wide (kernel 9) because NIR absorption features are broad humps tens of
    nanometres across, not sharp lines.
    """

    def __init__(self, epochs: int = 300, batch_size: int = 64, lr: float = 1e-3,
                 random_state: int = 0, verbose: bool = False, **_):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.random_state, self.verbose = random_state, verbose
        self.mu_ = self.sd_ = self.net_ = None

    @staticmethod
    def _require_torch():
        try:
            import torch  # noqa
            return torch
        except ImportError:
            raise ImportError(
                "the CNN baseline needs PyTorch.\n"
                "Fix:   pip install -r requirements-vision.txt\n"
                "The other baselines run without it."
            )

    def _build(self, torch, n_in):
        nn = torch.nn
        return nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), nn.Flatten(),
            nn.Dropout(0.1), nn.Linear(32 * 8, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def fit(self, X, y):
        torch = self._require_torch()
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu_) / self.sd_

        xt = torch.from_numpy(Xs).unsqueeze(1)
        yt = torch.from_numpy(y).unsqueeze(1)
        self.net_ = self._build(torch, X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = torch.nn.SmoothL1Loss(beta=0.02)

        n = len(xt)
        self.net_.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue        # BatchNorm needs >1 sample
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net_(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            sched.step()
            if self.verbose and ep % 50 == 0:
                print(f"    cnn epoch {ep:3d}  loss {tot / n:.5f}", flush=True)
        return self

    def predict(self, X):
        torch = self._require_torch()
        self.net_.eval()
        Xs = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        with torch.no_grad():
            out = self.net_(torch.from_numpy(Xs).unsqueeze(1))
        return out.numpy().ravel()
