FROM jupyter/scipy-notebook

RUN pip install \
    scikit-learn \
    geopandas \
    seaborn \
    ipywidgets \
    tensorflow \
    kneed \
    pillow \
    ipympl
