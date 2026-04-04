import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        # Initialize centroids randomly from the data points
        random_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_idxs]

        for i in range(self.max_iter):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([
                X[self.labels_ == j].mean(axis=0) if np.any(self.labels_ == j) else self.centroids[j]
                for j in range(self.n_clusters)
            ])

            # Check for convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Example usage:
if __name__ == "__main__":
    # Generate some random data
    X = np.random.rand(100, 2)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    print("Centroids:", kmeans.centroids)
    print("Labels:", kmeans.labels_)