FROM python:3.8-slim

ENV VIRTUAL_ENV=/opt/venv
RUN apt update && apt install build-essential -y --no-install-recommends
RUN python3 -m venv $VIRTUAL_ENV
RUN /opt/venv/bin/python3 -m pip install --upgrade pip
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
EXPOSE 8002/tcp

# Install dependencies
COPY requirements.txt . 
RUN pip install -r requirements.txt

# RUN the application
# VOLUME ./src/modelos
COPY src/main.py .
COPY src/classificador.py .
COPY src/manifestacao.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
# CMD ["ls", "-la", "modelos"]
