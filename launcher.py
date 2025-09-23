import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Sesión de SageMaker
sagemaker_session = sagemaker.Session()

# Rol de ejecución (toma el rol de la instancia actual)
role = sagemaker.get_execution_role()

# Bucket de S3 donde tengas el genetico
#bucket = "genetico"

sklearn_estimator = SKLearn(
    entry_point="genetic.py",  # El script de entrenamiento
    dependencies=["requirements.txt"],
    source_dir=".",  # subida de todo el código local
    #source_dir=f"s3://{bucket}/PyCatan/",  # Esto sería si tienes el código en S3
    role=role,
    instance_type="ml.t3.large",  # Tipo de instancia (puedes cambiarlo) medium no va
    instance_count=1,
    framework_version="0.23-1",  # Versión de SKLearn
    py_version="py3",
    sagemaker_session=sagemaker_session
)

sklearn_estimator.fit()