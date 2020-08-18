FROM jupyter/scipy-notebook

RUN pip install \
    scikit-learn \
    geopandas \
    seaborn \
    ipywidgets \
    tensorflow \
    tensorflow-datasets \
    kneed \
    pillow \
    ipympl \
    ipycanvas
