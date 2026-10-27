# syntax=docker/dockerfile:1
# Rebuildable legacy user space. No private/local parent image is required.
# Archives are downloaded from upstream distributors and checked before use.
# The upstream EMISAC source is mounted read-only at runtime, not patched.
FROM python:2.7.18-buster@sha256:d8fac68ebdc45b8d66d53f1ed6c1532da81109a8f5532a6ca0c951ed31107d70
RUN curl -fsSL https://repo.anaconda.com/archive/Anaconda-2.1.0-Linux-x86_64.sh -o /tmp/anaconda.sh \
    && echo '191fbf290747614929d0bdd576e330c944b22a67585d1c185e0d2b3a3e65e1c0  /tmp/anaconda.sh' | sha256sum -c - \
    && bash /tmp/anaconda.sh -b -p /opt/legacy \
    && ln -sf /opt/legacy/bin/python /usr/bin/python \
    && rm /tmp/anaconda.sh
ENV PATH=/opt/legacy/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
RUN /usr/bin/curl -fsSL https://files.pythonhosted.org/packages/22/82/c449ce5a49745d1f4cf6228d94b2f126e03f5316e4218f347db3e19bd1c0/joblib-0.8.3.tar.gz -o /tmp/joblib.tar.gz \
    && echo 'f163790e4c08b52c3c37db132c0528c3041da9e1d0095199bebd0c9876e1fcd6  /tmp/joblib.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/joblib.tar.gz -C /tmp \
    && cd /tmp/joblib-0.8.3 && python setup.py install \
    && rm /tmp/joblib.tar.gz
RUN /usr/bin/curl -fsSL https://files.pythonhosted.org/packages/71/e7/881fa2b6195141a2b91035b63ff78a6f11c5dbec5d39b012e9fadc193a95/scikit-image-0.9.3.tar.gz -o /tmp/skimage.tar.gz \
    && echo '2c29c65aacdfc056efd0a3b713b5dde666356ffb39e5e2bad3e0d6dbb62524b3  /tmp/skimage.tar.gz' | sha256sum -c - \
    && mkdir /tmp/skimage-build \
    && tar -xzf /tmp/skimage.tar.gz -C /tmp/skimage-build \
    && cd /tmp/skimage-build/scikit-image-0.9.3 \
    && python setup.py install --prefix=/opt/skimage093 --single-version-externally-managed --record=/tmp/skimage093-install.txt \
    && rm /tmp/skimage.tar.gz
ENV PYTHONPATH=/opt/skimage093/lib/python2.7/site-packages
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
WORKDIR /official
CMD ["python", "--version"]
