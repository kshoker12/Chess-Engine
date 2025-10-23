# Use AWS base image for Python 3.12
FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy code and model
COPY app.py ${LAMBDA_TASK_ROOT}
COPY engine.py ${LAMBDA_TASK_ROOT}
COPY features.py ${LAMBDA_TASK_ROOT}
COPY eval_model.py ${LAMBDA_TASK_ROOT}
COPY sk_eval.joblib ${LAMBDA_TASK_ROOT}

# Set handler
CMD [ "app.handler" ]