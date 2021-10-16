FROM pytorch/pytorch

# Change to a pytorch image

WORKDIR /src

COPY src .

# RUN mkdir data binaries

RUN apt-get update
RUN apt-get install unzip git -y

# Command requested by hugging face for pushing models to their remote repo
RUN git config --global credential.helper store

RUN pip install -r requirements.txt

CMD ["/bin/bash", "train.sh"]