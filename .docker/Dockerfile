FROM skrjtech/myfaceswap:latest

RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc
    
RUN set -x \
    && git clone https://github.com/skrjtech/linux-setup.git \
    && cd linux-setup \
    && . installer/pycharm.sh \
    && set +x
    
CMD [ "/bin/bash" ]