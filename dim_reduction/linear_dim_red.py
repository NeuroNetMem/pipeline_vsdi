# Dimensionality reduction
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline

def pca_ica(vsdi, mask):
    X = vsdi.transpose(2, 0, 1)
    X = X*mask
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])

    # Create a pipeline with PCA and ICA
    pipe = Pipeline([
        ('pca', PCA(n_components=10)),
        ('ica', FastICA(n_components=10, max_iter=200,
                        random_state=1, whiten='unit-variance'))
    ])

    out = pipe.fit(X)

    fingerprints = out.named_steps["ica"].components_ @ out.named_steps["pca"].components_
    timecourses = fingerprints @ X.T

    explained_variance = out.named_steps["pca"].explained_variance_ratio_

    return fingerprints, timecourses, explained_variance
