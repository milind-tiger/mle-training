FROM continuumio/anaconda

WORKDIR /MLE-TRAINING
COPY . . 
COPY env.yml .
RUN conda env create -f env.yml 

RUN conda init bash 

RUN echo 'conda activate mle_training'  > ~/.bashrc

CMD ["python" , "test.py"]
