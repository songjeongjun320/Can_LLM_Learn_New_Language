import kagglehub

# Download latest version
path = kagglehub.dataset_download("adiamaan/movie-subtitle-dataset")

print("Path to dataset files:", path)