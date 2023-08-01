# Use the Micromamba base image
FROM mambaorg/micromamba:latest

COPY environment.yml .

# Set the environment variables
ENV RESULTS_FOLDER_PATH=/app/data/processed
ENV RESOURCES_FOLDER_PATH=/app/resources
ENV DATASET_FOLDER_PATH=/app/data/external/segmented

RUN micromamba install -y -n base -f environment.yml && micromamba clean --all --yes

WORKDIR /app/

COPY src ./src
COPY run.py .
COPY resources ./resources

# Run the app.py script
CMD ["sh"]