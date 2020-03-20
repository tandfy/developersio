from urllib.parse import urlparse
import boto3
import time
import json
from datetime import datetime
from os import path
from typing import List
from logging import getLogger, StreamHandler, INFO
from pprint import pformat

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)

# ユーザが作成可能なデータセットタイプ(EVENT_INTERACTIONS(Events)データセットはEventTracker作成時に自動作成なので、含まない)
CREATABLE_DATASET_TYPES = ['Interactions', 'Items', 'Users']


def _wait(max_wait_seconds, status_getter_func):
    logger.info('waiting')
    max_time = time.time() + max_wait_seconds
    SLEEP_SECONDS = 10
    LOG_DOT_INTERVAL = 12
    i = 0
    while time.time() < max_time:
        status = status_getter_func()

        if status == 'ACTIVE':
            logger.info(status)
            return
        elif status == 'CREATE FAILED':
            raise RuntimeError(status)

        time.sleep(SLEEP_SECONDS)
        if i % LOG_DOT_INTERVAL == 0:
            logger.info('.')
        i += 1
    raise TimeoutError()


class S3:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def put_object(self, data, s3_uri):
        bucket, key = self.parse_s3_uri(s3_uri)
        return self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data
        )

    def get_object(self, s3_uri):
        bucket, key = self.parse_s3_uri(s3_uri)
        return self.s3_client.get_object(
            Bucket=bucket,
            Key=key
        )['Body'].read()

    @classmethod
    def parse_s3_uri(cls, s3_uri):
        parsed = urlparse(s3_uri)
        return parsed.netloc, parsed.path.lstrip('/')


class DatasetObject:
    def __init__(self, dataset_type, fields, data_location):
        if dataset_type not in CREATABLE_DATASET_TYPES:
            raise ValueError(
                f'You must specify valid dataset type, allowed values: {CREATABLE_DATASET_TYPES}')

        self.dataset_type = dataset_type
        self.fields = fields
        self.data_location = data_location


class EventObject:
    def __init__(self, event_type, properties, event_id=None, sent_at=datetime.now()):
        self.event_type = event_type
        self.properties = json.dumps(properties)
        self.event_id = event_id
        self.sent_at = sent_at

    def to_dict(self):
        event = {
            'eventType': self.event_type,
            'properties': self.properties,
            'sentAt': self.sent_at
        }
        if self.event_id:
            event['eventId'] = self.event_id
        return event


class EventTracker:
    def __init__(self, event_tracker_arn):
        self.personalize_client = boto3.client('personalize')
        self.personalize_events_client = boto3.client('personalize-events')
        self.event_tracker_arn = event_tracker_arn
        self.tracking_id = self.personalize_client.describe_event_tracker(
            eventTrackerArn=event_tracker_arn)['eventTracker']['trackingId']
        event_tracker = self.personalize_client.describe_event_tracker(
            eventTrackerArn=event_tracker_arn)['eventTracker']
        logger.info(pformat(event_tracker))

    def put_events(self, user_id, session_id, event_list: List[EventObject]):
        return self.personalize_events_client.put_events(
            trackingId=self.tracking_id,
            userId=str(user_id),
            sessionId=str(session_id),
            eventList=[event.to_dict() for event in event_list]
        )

    def delete(self):
        return self.personalize_client.delete_event_tracker(
            eventTrackerArn=self.event_tracker_arn)


class Campaign:
    def __init__(self, campaign_arn):
        self.campaign_arn = campaign_arn
        self.personalize_client = boto3.client('personalize')
        self.personalize_runtime_client = boto3.client('personalize-runtime')
        campaign = self.personalize_client.describe_campaign(
            campaignArn=campaign_arn)['campaign']
        logger.info(pformat(campaign))

    def get_recommendations(self, num_results, user_id=None, item_id=None, context=None):
        params = {
            'campaignArn': self.campaign_arn,
            'numResults': num_results
        }
        if context:
            params['context'] = context
        if user_id:
            params['userId'] = str(user_id)
        if item_id:
            params['itemId'] = str(item_id)
        if (not user_id) & (not item_id):
            raise ValueError('either user_id or item_id must be specified')
        items = self.personalize_runtime_client.get_recommendations(
            **params
        )['itemList']
        return [item['itemId'] for item in items]

    def get_recommendations_for_users(self, user_ids, num_results):
        assert len(user_ids) == len(set(user_ids))
        recommendations = {}
        for user_id in user_ids:
            recommendations[user_id] = self.get_recommendations(
                user_id=user_id,
                num_results=num_results
            )
        return recommendations

    def delete(self):
        return self.personalize_client.delete_campaign(campaignArn=self.campaign_arn)


class Solution:
    def __init__(self, solution_arn):
        self.solution_arn = solution_arn
        self.solution_name = path.basename(self.solution_arn)
        self.personalize_client = boto3.client('personalize')
        dataseg_group = self.personalize_client.describe_solution(
            solutionArn=solution_arn)['solution']
        logger.info(pformat(dataseg_group))

    def get_latest_solution_version_arn(self):
        return self.personalize_client.describe_solution(solutionArn=self.solution_arn)['solution']['latestSolutionVersion']['solutionVersionArn']

    def wait_for_batch_inference_job(self, job_arn, max_wait_seconds=60 * 60):
        logger.info(job_arn)
        _wait(
            max_wait_seconds,
            lambda: self.personalize_client.describe_batch_inference_job(
                batchInferenceJobArn=job_arn)['batchInferenceJob']['status']
        )

    def wait_for_creating_campaign(self, campaign_arn, max_wait_seconds=30 * 60):
        logger.info(campaign_arn)
        _wait(
            max_wait_seconds,
            lambda: self.personalize_client.describe_campaign(
                campaignArn=campaign_arn)['campaign']['status']
        )

    def wait_for_creating_solution_version(self, solution_version_arn, max_wait_seconds=2 * 60 * 60):
        logger.info(solution_version_arn)
        _wait(
            max_wait_seconds,
            lambda: self.personalize_client.describe_solution_version(
                solutionVersionArn=solution_version_arn)['solutionVersion']['status']
        )

    def create_solution_version(self, training_mode='FULL', wait=True, max_wait_seconds=2 * 60 * 60):
        solution_version_arn = self.personalize_client.create_solution_version(
            solutionArn=self.solution_arn,
            trainingMode=training_mode
        )['solutionVersionArn']
        if wait:
            self.wait_for_creating_solution_version(
                solution_version_arn, max_wait_seconds=max_wait_seconds)
        return solution_version_arn

    def create_campaign(self, min_provisioned_tps, solution_version_arn=None, wait=True):
        if not solution_version_arn:
            solution_version_arn = self.get_latest_solution_version_arn()
        solution_version_name = path.basename(solution_version_arn)
        campaign_arn = self.personalize_client.create_campaign(
            name=f'{self.solution_name}-{solution_version_name}',
            solutionVersionArn=solution_version_arn,
            minProvisionedTPS=min_provisioned_tps,
        )['campaignArn']

        if wait:
            self.wait_for_creating_campaign(campaign_arn)
        return Campaign(campaign_arn)

    def create_batch_inference_job(self, input_data_path, output_data_dir_path, num_results, role_arn, solution_version_arn=None, wait=True):
        if not solution_version_arn:
            solution_version_arn = self.get_latest_solution_version_arn()
        solution_version_name = path.basename(solution_version_arn)

        current_dt = datetime.now().strftime('%Y%m%d-%H%M%S')
        batch_inference_job_arn = self.personalize_client.create_batch_inference_job(
            jobName=f'{self.solution_name}-{solution_version_name}-{current_dt}',
            solutionVersionArn=solution_version_arn,
            numResults=num_results,
            jobInput={
                's3DataSource': {
                    'path': input_data_path
                }
            },
            jobOutput={
                's3DataDestination': {
                    'path': output_data_dir_path,
                }
            },
            roleArn=role_arn
        )['batchInferenceJobArn']

        if wait:
            self.wait_for_batch_inference_job(batch_inference_job_arn)
        return batch_inference_job_arn

    def batch_inference(self, s3_prefix, data, num_results, role_arn):
        input_data_path = path.join(s3_prefix, 'input.jsonl')
        output_data_dir_path = path.dirname(input_data_path) + '/'
        s3 = S3()
        s3.put_object(data, input_data_path)

        self.create_batch_inference_job(
            input_data_path, output_data_dir_path, num_results, role_arn)
        output_data_path = input_data_path + '.out'
        body = s3.get_object(output_data_path)
        return [json.loads(ss) for ss in body.splitlines()]

    def batch_inference_for_users(self, s3_prefix, user_ids: List[int], num_results, role_arn):
        assert len(user_ids) == len(set(user_ids))

        data = '\n'.join(
            [json.dumps({'userId': str(user_id)}) for user_id in user_ids])
        recommendations = self.batch_inference(
            s3_prefix, data, num_results, role_arn)

        recommendations = dict(
            [self._transform_batch_inferenced_data(x) for x in recommendations])
        return recommendations

    def fetch_batch_inferenced_recommendations_for_users(self, batch_inference_job_arn):
        job = self.personalize_client.describe_batch_inference_job(
            batchInferenceJobArn=batch_inference_job_arn
        )['batchInferenceJob']

        output_data_path = path.join(
            job['jobOutput']['s3DataDestination']['path'],
            path.basename(job['jobInput']['s3DataSource']['path']) + '.out'
        )

        body = S3().get_object(output_data_path)
        with open('test.jsonl', 'wb') as f:
            f.write(body)

        recommendations = dict([self._transform_batch_inferenced_data(
            json.loads(ss)) for ss in body.splitlines()])
        return recommendations

    @classmethod
    def _transform_batch_inferenced_data(self, dic):
        return (
            int(dic['input']['userId']),
            list(map(lambda x: int(x), dic['output']['recommendedItems']))
        )


class DatasetGroup:
    def __init__(self, dataset_group_arn):
        self.personalize_client = boto3.client('personalize')
        self.dataset_group_arn = dataset_group_arn
        self.dataset_group_name = path.basename(dataset_group_arn)
        dataseg_group = self.personalize_client.describe_dataset_group(
            datasetGroupArn=dataset_group_arn)['datasetGroup']
        logger.info(pformat(dataseg_group))

    def wait_for_creating_dataset_group(self, max_wait_seconds=10 * 60):
        logger.info(self.dataset_group_arn)
        _wait(
            max_wait_seconds,
            lambda: self.personalize_client.describe_dataset_group(
                datasetGroupArn=self.dataset_group_arn)['datasetGroup']['status']
        )

    def wait_for_importing_data(self, job_arn, max_wait_seconds=60 * 60):
        logger.info(job_arn)
        _wait(
            max_wait_seconds,
            lambda: self.personalize_client.describe_dataset_import_job(
                datasetImportJobArn=job_arn)['datasetImportJob']['status']
        )

    def wait_for_creating_event_tracker(self, event_tracker_arn, max_wait_seconds=10 * 60):
        logger.info(event_tracker_arn)
        _wait(
            max_wait_seconds,
            lambda: self.personalize_client.describe_event_tracker(
                eventTrackerArn=event_tracker_arn)['eventTracker']['status']
        )

    def list_solution_arns(self):
        return [solution['solutionArn'] for solution in self.personalize_client.list_solutions(
            datasetGroupArn=self.dataset_group_arn)['solutions']]

    def delete_all_campaings(self):
        solution_arns = self.list_solution_arns()
        for solution_arn in solution_arns:
            campaigns = self.personalize_client.list_campaigns(
                solutionArn=solution_arn)['campaigns']
            for campaign in campaigns:
                logger.info(f'deleting {campaign["campaignArn"]}')
                self.personalize_client.delete_campaign(
                    campaignArn=campaign['campaignArn'])

        for solution_arn in solution_arns:
            while len(self.personalize_client.list_campaigns(solutionArn=solution_arn)['campaigns']):
                time.sleep(5)

    def delete_dataset_by_type(self, dataset_type):
        dataset = self.describe_dataset_by_type(dataset_type)
        dataset_arn = dataset['datasetArn']
        logger.info(f'deleting {dataset_arn}')
        self.personalize_client.delete_dataset(datasetArn=dataset_arn)

        while True:
            try:
                self.personalize_client.describe_dataset(
                    datasetArn=dataset_arn)
            except self.personalize_client.exceptions.ResourceNotFoundException:
                break

            time.sleep(5)

    def delete_dataset_group(self, force=False):

        if not force:
            self.personalize_client.delete_dataset_group(
                datasetGroupArn=self.dataset_group_arn)
            return

        # データセットグループ配下のリソースを順番に削除

        # キャンペーン
        self.delete_all_campaings()

        # イベントトラッカー
        for event_tracker in self.personalize_client.list_event_trackers(datasetGroupArn=self.dataset_group_arn)['eventTrackers']:
            logger.info(f'deleting {event_tracker["eventTrackerArn"]}')
            self.personalize_client.delete_event_tracker(
                eventTrackerArn=event_tracker['eventTrackerArn'])
        while len(self.personalize_client.list_event_trackers(datasetGroupArn=self.dataset_group_arn)['eventTrackers']):
            time.sleep(5)

        # ソリューション
        for solution_arn in self.list_solution_arns():
            logger.info(f'deleting {solution_arn}')
            self.personalize_client.delete_solution(solutionArn=solution_arn)
        while len(self.list_solution_arns()):
            time.sleep(5)

        # データセット
        for dataset in self.personalize_client.list_datasets(datasetGroupArn=self.dataset_group_arn)['datasets']:
            logger.info(f'deleting {dataset["datasetArn"]}')
            self.personalize_client.delete_dataset(
                datasetArn=dataset['datasetArn'])
        while len(self.personalize_client.list_datasets(datasetGroupArn=self.dataset_group_arn)['datasets']):
            time.sleep(5)

        # データセットグループ
        logger.info(f'deleting {self.dataset_group_arn}')
        self.personalize_client.delete_dataset_group(
            datasetGroupArn=self.dataset_group_arn)

    def create_schema(self, dataset_type, fields, use_it_if_existed=False):
        schema_name = f'{self.dataset_group_name}-{dataset_type}'
        schema = json.dumps({
            'type': 'record',
            'name': dataset_type,
            'namespace': 'com.amazonaws.personalize.schema',
            'fields': fields,
            'version': '1.0'
        })
        try:
            schema_arn = self.personalize_client.create_schema(
                name=schema_name,
                schema=schema
            )['schemaArn']
        except self.personalize_client.exceptions.ResourceAlreadyExistsException as e:
            if not use_it_if_existed:
                raise e

            found_schema = self.describe_schema_by_name(schema_name)

            if schema != found_schema['schema']:
                raise e

            schema_arn = found_schema['schemaArn']
            logger.info(
                'did not create schema because same schema already existed')
        return schema_arn

    def describe_schema_by_name(self, schema_name):

        account_id = boto3.client('sts').get_caller_identity()['Account']
        region = self.personalize_client.meta.region_name
        schema_arn = f'arn:aws:personalize:{region}:{account_id}:schema/{schema_name}'
        return self.personalize_client.describe_schema(schemaArn=schema_arn)['schema']

    def describe_dataset_by_type(self, dataset_type):
        account_id = boto3.client('sts').get_caller_identity()['Account']
        region = self.personalize_client.meta.region_name
        if dataset_type.lower() == 'events':
            dataset_type = 'EVENT_INTERACTIONS'
        dataset_arn = f'arn:aws:personalize:{region}:{account_id}:dataset/{self.dataset_group_name}/{dataset_type.upper()}'
        return self.personalize_client.describe_dataset(datasetArn=dataset_arn)['dataset']

    def create_dataset(self, dataset_type, schema_arn, use_it_if_existed=False):
        try:
            dataset_arn = self.personalize_client.create_dataset(
                name=f'{self.dataset_group_name}-{dataset_type}',
                datasetType=dataset_type,
                datasetGroupArn=self.dataset_group_arn,
                schemaArn=schema_arn
            )['datasetArn']
        except self.personalize_client.exceptions.ResourceAlreadyExistsException as e:
            if not use_it_if_existed:
                raise e

            found_dataset = self.describe_dataset_by_type(dataset_type)

            if schema_arn != found_dataset['schemaArn']:
                raise e

            dataset_arn = found_dataset['datasetArn']
            logger.info(
                'did not create dataset because same dataset already existed')
        return dataset_arn

    def create_importing_data_job(self, dataset_arn, data_location, role_arn):

        current_dt = datetime.now().strftime('%Y%m%d-%H%M%S')
        dataset_type = path.basename(dataset_arn)
        return self.personalize_client.create_dataset_import_job(
            jobName=f'{self.dataset_group_name}-{dataset_type}-{current_dt}',
            datasetArn=dataset_arn,
            dataSource={
                'dataLocation': data_location
            },
            roleArn=role_arn
        )['datasetImportJobArn']

    def create_datasets_and_import_data(self, datasets: List[DatasetObject], role_arn, use_it_if_existed=False):
        dataset_import_job_arns = []
        for dataset in datasets:
            dataset_type = dataset.dataset_type
            logger.info(f'creating dataset:{dataset_type}')

            # スキーマ作成
            schema_arn = self.create_schema(
                dataset_type=dataset_type,
                fields=dataset.fields,
                use_it_if_existed=use_it_if_existed
            )

            # データセット作成
            dataset_arn = self.create_dataset(
                dataset_type=dataset_type,
                schema_arn=schema_arn,
                use_it_if_existed=use_it_if_existed)

            # データ読み込み
            job_arn = self.create_importing_data_job(
                dataset_arn=dataset_arn,
                data_location=dataset.data_location,
                role_arn=role_arn
            )
            dataset_import_job_arns.append(job_arn)

        for job_arn in dataset_import_job_arns:
            self.wait_for_importing_data(job_arn)

    def create_event_tracker(self, wait=True):
        event_tracker_arn = self.personalize_client.create_event_tracker(
            name=self.dataset_group_name,
            datasetGroupArn=self.dataset_group_arn
        )['eventTrackerArn']

        if wait:
            self.wait_for_creating_event_tracker(event_tracker_arn)
        return EventTracker(event_tracker_arn)

    def create_solution(self, recipe_arn=None, name_suffix='', basename=None, **kwargs):
        if not basename:
            basename = self.dataset_group_name
            if recipe_arn:
                basename += '-' + path.basename(recipe_arn)

        name = basename + name_suffix
        solution_arn = self.personalize_client.create_solution(
            datasetGroupArn=self.dataset_group_arn,
            name=name,
            recipeArn=recipe_arn,
            **kwargs
        )['solutionArn']
        return Solution(solution_arn)

    @classmethod
    def create_dataset_group(cls, dataset_group_name):
        personalize_client = boto3.client('personalize')
        dataset_group_arn = personalize_client.create_dataset_group(
            name=dataset_group_name)['datasetGroupArn']
        dataset_group = DatasetGroup(dataset_group_arn)
        dataset_group.wait_for_creating_dataset_group()
        return dataset_group
