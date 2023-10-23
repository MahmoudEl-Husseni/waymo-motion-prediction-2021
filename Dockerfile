FROM nvcr.io/nvidia/pytorch:23.08-py3
ADD . .
RUN pip install -r requirements.txt
CMD ["sh", "script.sh"]