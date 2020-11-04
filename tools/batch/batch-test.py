import subprocess
import boto3

batch = boto3.client('batch', region_name='us-east-1')
response = batch.describe_job_definitions(status='ACTIVE')['jobDefinitions']
instance_type_info = {}
for res in response:
    jobDefinition = res['jobDefinitionName'] # example: gluon-cv-p2_8xlarge:1
    instance = jobDefinition.split('-')[-1].split(':')[0].replace('large', '') # example: p2_8x
    job_queue = jobDefinition.split('-')[-1].split('_')[0] # example: p2
    instance_type_info[instance] = {'job_definition': jobDefinition, 'job_queue': job_queue}

for instance in instance_type_info:
    command = ['python3', \
               'submit-job.py', \
                '--name', instance+'-test', \
                '--job-type', instance.replace('large', ''), \
                '--source-ref', 'master', \
                '--work-dir', 'docs/tutorials/classification', \
                '--remote', 'https://github.com/dmlc/gluon-cv', \
                '--command', 'python3 demo_cifar10.py'
              ]
    subprocess.run(command)
    