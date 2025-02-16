from sklearn.mixture import GaussianMixture

# Giả sử X_train là tập dữ liệu đặc trưng của một người
gmm = GaussianMixture(n_components=16, covariance_type='diag')
gmm.fit(X_train)