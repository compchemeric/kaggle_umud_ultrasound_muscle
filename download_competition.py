import kagglehub


# Download latest version
path = kagglehub.competition_download(
    "umud-challenge-muscle-architecture-in-ultrasound-data"
)

print("Path to competition files:", path)
