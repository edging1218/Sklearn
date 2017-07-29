from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#StandardScaler is to standardize each features (eg, fish with weights, length, aspect ratio; or appartment with area, #bathroom)
scaler = StandardScaler()
kmeans = KMeans(n_clusters = 4)
pipeline = make_pipeline(scaler, kmeans)


pipeline.fit(samples)
labels = pipeline.predict(samples)

# Normalizer is to standardize each entry/row (eg, stock price for each company as the rows and dates as the columns)
# Import Normalizer
# from sklearn.preprocessing import Normalizer
# Create a normalizer: normalizer
# normalizer = Normalizer()

