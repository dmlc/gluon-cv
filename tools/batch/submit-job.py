import argparse
import random
import re
import sys
import time
from datetime import datetime

import boto3
from botocore.compat import total_seconds


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--profile', help='profile name of aws account.', type=str,
                    default=None)
parser.add_argument('--region', help='Default region when creating new connections', type=str,
                    default=None)
parser.add_argument('--name', help='name of the job', type=str, default='dummy')
parser.add_argument('--job-type', help='type of job to submit.', type=str,
                    default='g4dn.4x')
parser.add_argument('--source-ref',
                    help='ref in GluonCV main github. e.g. master, refs/pull/500/head',
                    type=str, default='master')
parser.add_argument('--work-dir',
                    help='working directory inside the repo. e.g. scripts/classification',
                    type=str, default='scripts/classification')
parser.add_argument('--saved-output',
                    help='output to be saved, relative to working directory. '
                         'it can be either a single file or a directory',
                    type=str, default='None')
parser.add_argument('--save-path',
                    help='s3 path where files are saved.',
                    type=str, default='batch/temp/{}'.format(datetime.now().isoformat()))
parser.add_argument('--command', help='command to run', type=str,
                    default='')
parser.add_argument('--remote',
                    help='git repo address. https://github.com/dmlc/gluon-cv',
                    type=str, default="https://github.com/dmlc/gluon-cv")
parser.add_argument('--wait', help='block wait until the job completes. '
                    'Non-zero exit code if job fails.', action='store_true')
parser.add_argument('--timeout', help='job timeout in seconds', default=None, type=int)


args = parser.parse_args()

session = boto3.Session(profile_name=args.profile, region_name=args.region)
batch, cloudwatch = [session.client(service_name=sn) for sn in ['batch', 'logs']]

response = batch.describe_job_definitions(status='ACTIVE')['jobDefinitions']
job_to_queue_map = {}
for res in response:
    jobDefinition = res['jobDefinitionName'] # example: gluon-cv-p2_8xlarge:1
    instance = jobDefinition.split('-')[-1].split(':')[0].replace('large', '') # example: p2_8x
    job_queue = jobDefinition.split('-')[-1].split('_')[0] # example: p2
    job_to_queue_map[instance] = {'job_definition': jobDefinition, 'job_queue': job_queue}
# ci is a special case because the name of the job doesn't match any instance type
# it uses job description g4dn_4x and job queue ci underneath
job_to_queue_map['ci'] = {'job_definition': job_to_queue_map['g4dn_4x']['job_definition'], 'job_queue': 'ci'}

def printLogs(logGroupName, logStreamName, startTime):
    kwargs = {'logGroupName': logGroupName,
              'logStreamName': logStreamName,
              'startTime': startTime,
              'startFromHead': True}

    lastTimestamp = 0
    while True:
        logEvents = cloudwatch.get_log_events(**kwargs)

        for event in logEvents['events']:
            lastTimestamp = event['timestamp']
            timestamp = datetime.utcfromtimestamp(lastTimestamp / 1000.0).isoformat()
            print('[{}] {}'.format((timestamp + '.000')[:23] + 'Z', event['message']))

        nextToken = logEvents['nextForwardToken']
        if nextToken and kwargs.get('nextToken') != nextToken:
            kwargs['nextToken'] = nextToken
        else:
            break
    return lastTimestamp


def nowInMillis():
    endTime = long(total_seconds(datetime.utcnow() - datetime(1970, 1, 1))) * 1000
    return endTime


def main():
    spin = ['-', '/', '|', '\\', '-', '/', '|', '\\']
    logGroupName = '/aws/batch/job'

    jobName = re.sub('[^A-Za-z0-9_\-]', '', args.name)[:128]  # Enforce AWS Batch jobName rules
    jobType = args.job_type.replace('.', '_')
    # check valid job type
    while jobType not in job_to_queue_map:
        instance_choices = job_to_queue_map.keys()
        print("Please provide a valid job type. Choices are:")
        for choice in instance_choices:
            print(choice.replace('_', '.'))
        jobType = input("Your job type (ctrl+d to exit): ").replace('.', '_')
    jobQueue = job_to_queue_map[jobType]['job_queue']
    jobDefinition = job_to_queue_map[jobType]['job_definition']
    wait = args.wait

    parameters = {
        'SOURCE_REF': args.source_ref,
        'WORK_DIR': args.work_dir,
        'SAVED_OUTPUT': args.saved_output,
        'SAVE_PATH': args.save_path,
        'COMMAND': args.command,
        'REMOTE': args.remote
    }
    kwargs = dict(
        jobName=jobName,
        jobQueue=jobQueue,
        jobDefinition=jobDefinition,
        parameters=parameters,
    )
    if args.timeout is not None:
        kwargs['timeout'] = {'attemptDurationSeconds': args.timeout}
    submitJobResponse = batch.submit_job(**kwargs)

    jobId = submitJobResponse['jobId']
    print('Submitted job [{} - {}] to the job queue [{}]'.format(jobName, jobId, jobQueue))

    spinner = 0
    running = False
    status_set = set()
    startTime = 0
    logStreamName = None
    while wait:
        time.sleep(random.randint(5, 10))
        describeJobsResponse = batch.describe_jobs(jobs=[jobId])
        status = describeJobsResponse['jobs'][0]['status']
        if status == 'SUCCEEDED' or status == 'FAILED':
            print('Output [{}]:\n {}'.format(logStreamName, '=' * 80))
            if logStreamName:
                startTime = printLogs(logGroupName, logStreamName, startTime) + 1
            print('=' * 80)
            print('Job [{} - {}] {}'.format(jobName, jobId, status))
            sys.exit(status == 'FAILED')

        elif status == 'RUNNING':
            logStreamName = describeJobsResponse['jobs'][0]['container']['logStreamName']
            if not running:
                running = True
                print('\rJob [{}, {}] is RUNNING.'.format(jobName, jobId))
                # if logStreamName:
            # if logStreamName:
            #     startTime = printLogs(logGroupName, logStreamName, startTime) + 1
            print('\rJob [{}, {}] is still RUNNING.'.format(jobName, jobId))
        elif status not in status_set:
            status_set.add(status)
            print('\rJob [%s - %s] is %-9s... %s' % (jobName, jobId, status, spin[spinner % len(spin)]),)
            sys.stdout.flush()
            spinner += 1


if __name__ == '__main__':
    main()
