# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

r"""A dag definition to train and deploy higgs model in Google Cloud ML Engine.

The ML model is based on Tensorflow Gradient Boosting Decision Trees algorithm,
namely TF boosted_trees, and the code is published on
https://github.com/tensorflow/models/tree/master/official/boosted_trees

Here, the dag downloads the data, trains the model and deploys it to
Google Cloud ML Engine.


# Set up.

Since this dag requires recent updates to some airflow operators, it works with
the airflow version >= 2.0.0 (or master branch synced later than 2018-04-12).

Use Cloud Composer or install master (HEAD) version of airflow like:
(maybe this part doesn't go to github, in favor of Cloud Composer)

```
$ sudo apt-get install python-dev build-essential virtualenv
$ virtualenv --system-site-packages ~/airflow-git
$ . ~/airflow-git/bin/activate
$ pip install apache-airflow>=2.0.0  # only after 2.0 is released, or below.
  # pip install git+https://github.com/apache/incubator-airflow.git#egg=apache-airflow[gcp_api]
$ airflow initdb
$ airflow scheduler  # add -D to run as a daemon
$ airflow webserver -p 8080  # add -D to run as a daemon
$ wget -P ~/airflow/dags/ https://github.com/apache/incubator-airflow/raw/master/airflow/contrib/example_dags/example_mlengine_for_higgs.py
```

To configure the pipeline, set these `Variables` in the UI under `Admin` menu:

 - gcp_region: the region to call ML Engine with. (e.g. us-east1)
 - gcp_project: the name of the project.
 - gcs_bucket: the name of the existing GCS bucket to put the data in.
     (without gs:// prefix)
 - hyperparameters (optional): in the form of arguments that the trainer
     receives.  e.g. --n_trees=100 --max_depth=6
 - official_model_version (optional): the official tensorflow model version
     number released at https://github.com/tensorflow/models/releases.
     By default, 1.9.0. boosted_trees model started to appear at 1.9.0.
 - mlengine_runtime_version (optional): the runtime version to use in ML Engine.
     It should be >= 1.8 to train boosted_trees, and by default, set as 1.8.

These `Connections` might be filled by default if using Cloud Composer, but if
not, please fill them under `Admin` menu in UI:

 - google_cloud_default:
      Conn_Type="Google Cloud Platform",
      Project_Id=<your project>
 - google_cloud_storage_default: the same as google_cloud_default.

NOTE: gcp_region would be better to be aligned with the region that the airflow
service is running. gcp_project and gcs_bucket should be created before
launching this workflow. gcs_bucket should be either multi-regional or regional
within the specified gcp_region.


# Status Monitoring

ML Engine jobs and models can be inspected in Google Cloud Console:

 - http://console.cloud.google.com/mlengine/jobs and
 - http://console.cloud.google.com/mlengine/models

respectively.


# Prediction

After a version is successfully created, you can query the server with own
data (two instances below).

```
$ GCP_PROJECT=<your_gcp_project>
$ MLENGINE_MODEL_VERSION=<your_version_name>
$ curl -H "Content-Type: application/json" \
    -H "Authorization: Bearer `gcloud auth print-access-token`" \
    -X POST https://ml.googleapis.com/v1/projects/${GCP_PROJECT}/models/boosted_trees_higgs/versions/${MLENGINE_MODEL_VERSION}:predict \
    -d '{"instances": ["0.869293,-0.635082,0.225690,0.327470,-0.689993,0.754202,-0.248573,-1.092064,0.0,1.374992,-0.653674,0.930349,1.107436,1.138904,-1.578198,-1.046985,0.0,0.657930,-0.010455,-0.045767,3.101961,1.353760,0.979563,0.978076,0.920005,0.721657,0.988751,0.876678", "1.595839,-0.607811,0.007075,1.818450,-0.111906,0.847550,-0.566437,1.581239,2.173076,0.755421,0.643110,1.426367,0.0,0.921661,-1.190432,-1.615589,0.0,0.651114,-0.654227,-1.274345,3.101961,0.823761,0.938191,0.971758,0.789176,0.430553,0.961357,0.957818"]}'
```

Replace `<your_gcp_project>` with your project name set in airflow, and find
`<your_version_name>` from the Google Cloud Console
(http://console.cloud.google.com/mlengine/models), in the form
of `version_<datetime>_<numbers>`.

The result would be something like (one prediction per line):

```
CLASSES       SCORES
[u'0', u'1']  [0.3559727072715759, 0.6440272927284241]
[u'0', u'1']  [0.8909763097763062, 0.10902369767427444]
```

This means the first instance has the probability of being a higgs process (i.e.
class `1`) is about 0.644, and the second instance has about 0.109 (or not being
-- class `0` -- with the probability of 0.891).


# Clean up

Unused versions/models can be cleaned up; though versions/models in Google Cloud
ML Engine are not charged as long as no prediction requests are made.
GCS bucket might be charged beyond free-tier usage, so please consider clean-up.

For details on pricing, refer to https://cloud.google.com/ml-engine/docs/pricing
and https://cloud.google.com/storage/pricing

If you're using an airflow service on Cloud, you might consider shutting it down
as well (either a managed airflow service or a VM running one) after use to save
cost.


# References

 - Tensorflow: https://tensorflow.org
 - Google Cloud ML Engine: https://cloud.google.com/ml-engine
 - Airflow ML Engine operators: http://airflow.readthedocs.io/en/latest/integration.html#cloud-ml-engine
 - BoostedTrees model: https://github.com/tensorflow/models/tree/master/official/boosted_trees

"""

import datetime
import os

from airflow import models
from airflow.contrib.hooks.gcs_hook import GoogleCloudStorageHook
from airflow.contrib.operators import mlengine_operator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.utils import trigger_rule

OFFICIAL_MODEL_VERSION = models.Variable.get('official_model_version', '1.9.0')
# The file name downloaded from https://github.com/tensorflow/models/releases
OFFICIAL_SRC_TARBALL = 'v{}.tar.gz'.format(OFFICIAL_MODEL_VERSION)
TRAINER_TARBALL_PREFIX = 'tensorflow_models_official'

yesterday = datetime.datetime.combine(
    datetime.datetime.today() - datetime.timedelta(1),
    datetime.datetime.min.time())

default_dag_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    # Setting start date as yesterday starts the DAG immediately when it is
    # detected in the Cloud Storage bucket.
    'start_date': yesterday,
    # To email on failure or retry set 'email' arg to your email and enable
    # emailing here.
    'email_on_failure': False,
    'email_on_retry': False,
    # Retry once, after waiting for 5 minutes for any failures.
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    # Google Cloud related settings (GCS/GCP/ML Engine).
    'gcs_bucket': models.Variable.get('gcs_bucket'),
    'gcp_project': models.Variable.get('gcp_project'),
    'gcp_region': models.Variable.get('gcp_region'),
    'model_name': 'boosted_trees_higgs',
    # Model related settings. runtime must be >= 1.8 for boosted_trees.
    'runtime_version': models.Variable.get('mlengine_runtime_version', '1.8'),
    'hyperparameters': models.Variable.get('hyperparameters', default_var='')
}

# Custom-built Cloud ML Engine package requires setup.py.
# https://cloud.google.com/ml-engine/docs/tensorflow/packaging-trainer#building_your_trainer_package_manually
# for details.
TRAINER_SETUP_PY = """
from setuptools import find_packages
from setuptools import setup
try: # for pip >= 10
  from pip._internal.download import PipSession
  from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
  from pip.download import PipSession
  from pip.req import parse_requirements
setup(
  name="{trainer_package_name}",
  version="{official_model_version}",
  # Extract existing requirements in the tensorflow official model.
  # Alternatively, you could just explicitly list only required ones.
  install_requirements=[
    str(req.req) for req in parse_requirements(
       "official/requirements.txt", session=PipSession())
  ],
  packages=find_packages(include=['official', 'official.*']),
  zip_safe=True,
  package_data=dict(official=['requirements.txt']),
  description="Tensorflow official models.",
)
""".format(trainer_package_name=TRAINER_TARBALL_PREFIX,
           official_model_version=OFFICIAL_MODEL_VERSION)

CREATE_TRAINER_PACKAGE_BASH_TEMPLATE = """
set -e
WORKDIR="$(mktemp -d)"
cd $WORKDIR

# The URL/tarball name can be found from the Release page.
wget https://github.com/tensorflow/models/archive/{src_tarball}

# Exclude the first directory, which is tensorflow-models-[hash]/
tar xvzf {src_tarball} --strip-components=1

# Add setup.py file as defined above.
cat << EOF > setup.py
{trainer_setup_py}
EOF

# Repackage a tarball then upload it to GCS.
python setup.py sdist
gsutil cp dist/{trainer_tarball} gs://{gcs_bucket}/

# Clean up.
rm -rf $WORKDIR
"""

# savedmodel is exported with the timestamp in its path; we want to export the
# latest model so here we extract the number from the path.
EXTRACT_LATEST_EXPORT_BASH_TEMPLATE = r"""
set -e
TIMESTAMP="$(gsutil ls {export_uri}/[0-9]*/saved_model.pb | sort | tail -1 |
    sed 's@{export_uri}/\([0-9]*\)/saved_model.pb@\1@')"
echo $TIMESTAMP
"""


def make_branch_by_gcs_exists_fn(obj_path, exists_task_id, non_exists_task_id):
  """Returns a function to branch by gcs object existence."""
  def data_check_fn():
    if GoogleCloudStorageHook().exists(
        default_dag_args['gcs_bucket'], obj_path):
      return exists_task_id
    return non_exists_task_id
  return data_check_fn


with models.DAG(
    'mlengine_for_higgs',
    # Continue to run DAG once per day
    schedule_interval=datetime.timedelta(days=1),
    default_args=default_dag_args) as dag:
  # Using custom ts, since {{ ts }} has incompatible characters to ML Engine.
  # Also adding uuid afterwards to allow retry.
  ts_nospecial = '{{ execution_date.strftime("%Y%m%dT%H%M%S") }}'
  uuid = '{{ macros.uuid.uuid4().hex[:8] }}'

  trainer_tarball = '{}-{}.tar.gz'.format(TRAINER_TARBALL_PREFIX,
                                          OFFICIAL_MODEL_VERSION)
  skip_create_trainer_package_op = DummyOperator(
      task_id='skip_create_trainer_package')
  create_trainer_package_op = BashOperator(
      task_id='create_trainer_package',
      bash_command=CREATE_TRAINER_PACKAGE_BASH_TEMPLATE.format(
          src_tarball=OFFICIAL_SRC_TARBALL,
          trainer_tarball=trainer_tarball,
          trainer_setup_py=TRAINER_SETUP_PY.strip(),
          gcs_bucket=default_dag_args['gcs_bucket']))
  join_after_create_package_op = DummyOperator(
      task_id='join_after_create_package',
      trigger_rule=trigger_rule.TriggerRule.ONE_SUCCESS)
  if_trainer_package_exists_op = BranchPythonOperator(
      task_id='if_trainer_package_exists',
      python_callable=make_branch_by_gcs_exists_fn(
          obj_path=trainer_tarball,
          exists_task_id=skip_create_trainer_package_op.task_id,
          non_exists_task_id=create_trainer_package_op.task_id))

  data_uri = os.path.join('gs://', default_dag_args['gcs_bucket'], 'data')
  model_uri = os.path.join('gs://', default_dag_args['gcs_bucket'], 'model',
                           ts_nospecial)
  export_uri = os.path.join('gs://', default_dag_args['gcs_bucket'], 'export',
                            ts_nospecial)

  def create_training_op(task_id, module, args):
    return mlengine_operator.MLEngineTrainingOperator(
        task_id=task_id,
        project_id=default_dag_args['gcp_project'],
        job_id='boosted_trees_{}_{}_{}'.format(task_id, ts_nospecial, uuid),
        package_uris=[
            os.path.join('gs://', default_dag_args['gcs_bucket'],
                         trainer_tarball)
        ],
        training_python_module=module,
        training_args=args,
        region=default_dag_args['gcp_region'],
        runtime_version=default_dag_args['runtime_version'])

  # data_download is using MLEngineTraining API with the same package/settings
  # but different module and args.
  # Download the data only when not existing already.
  skip_data_download_op = DummyOperator(task_id='skip_data_download')
  data_download_op = create_training_op(
      task_id='data_download',
      module='official.boosted_trees.data_download',
      args=['--data_dir={}'.format(data_uri)])
  join_after_download_op = DummyOperator(
      task_id='join_after_download',
      trigger_rule=trigger_rule.TriggerRule.ONE_SUCCESS)
  if_data_exists_op = BranchPythonOperator(
      task_id='if_data_exists',
      python_callable=make_branch_by_gcs_exists_fn(
          obj_path=os.path.join('data', 'HIGGS.csv.gz.npz'),
          exists_task_id=skip_data_download_op.task_id,
          non_exists_task_id=data_download_op.task_id))

  training_op = create_training_op(
      task_id='training',
      module='official.boosted_trees.train_higgs',
      args=[
          '--data_dir={}'.format(data_uri),
          '--model_dir={}'.format(model_uri),
          '--export_dir={}'.format(export_uri),
      ] + default_dag_args['hyperparameters'].split())

  # export_savedmodel exports to the path with the timestamp, thus
  # it's required to extract info on which is the latest one.
  extract_latest_export_op = BashOperator(
      task_id='extract_latest_export',
      xcom_push=True,
      bash_command=EXTRACT_LATEST_EXPORT_BASH_TEMPLATE.format(
          export_uri=export_uri))
  export_timestamp = ('{{ task_instance.xcom_pull(task_ids="%s") }}' %
                      extract_latest_export_op.task_id)

  create_version_op = mlengine_operator.MLEngineVersionOperator(
      task_id='create_version',
      project_id=default_dag_args['gcp_project'],
      model_name=default_dag_args['model_name'],
      version={
          'name': 'version_{}_{}'.format(ts_nospecial, export_timestamp),
          'deploymentUri': os.path.join(export_uri, export_timestamp),
          'runtimeVersion': default_dag_args['runtime_version'],
      },
      operation='create')

  # if_data_exists_op makes the branches, then the workflow is a one-way.
  (if_trainer_package_exists_op >>
   [create_trainer_package_op, skip_create_trainer_package_op] >>
   join_after_create_package_op >> if_data_exists_op >>
   [data_download_op, skip_data_download_op] >> join_after_download_op >>
   training_op >> extract_latest_export_op >> create_version_op)
