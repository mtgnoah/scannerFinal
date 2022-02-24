FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt ./
COPY scanner/ ./
RUN ls -la ./

RUN python3.9 -m pip install -r requirements.txt -t .
RUN python3.9 -m pip install --upgrade google-cloud-vision -t .


CMD ["detect.lambda_handler"]
