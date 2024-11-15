FROM dustynv/tritonserver:r35.4.1

RUN wget -O /tmp/boost.tar.gz \
    https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
    && (cd /tmp && tar xzf boost.tar.gz) && cd /tmp/boost_1_80_0 && ./bootstrap.sh \
    && ./b2 --with-filesystem && cp /tmp/boost_1_80_0/stage/lib/libboost_filesystem.so.1.80.0 /usr/lib

CMD ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models", "--log-verbose=1"]
