FROM python:3.9-slim
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY /stuff/ /deploy/
WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
