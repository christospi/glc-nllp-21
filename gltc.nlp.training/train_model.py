import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import click
import  logging
from nomothesia_nlp.experiments.configurations.configuration import Configuration
from nomothesia_nlp.experiments import NamedEntityRecognition
from nomothesia_nlp.experiments import EuroVocClassification
from nomothesia_nlp.experiments import RaptarchisClassification
cli = click.Group()
logging.getLogger('transformers').setLevel(logging.ERROR)


@cli.command()
@click.option('--task_type', default='text_classification')
@click.option('--task_name', default='raptarchis_classification')
def run(task_type, task_name):
    Configuration.configure(task_type, task_name)

    if task_name == 'named_entity_recognition':
        experiment = NamedEntityRecognition()
    elif task_name == 'eurovoc_classification':
        experiment = EuroVocClassification()
    elif task_name == 'raptarchis_classification':
        experiment = RaptarchisClassification()
    else:
        raise Exception('Task type "{}" is not supported'.format(task_type))
    experiment.run_operation()


if __name__ == '__main__':
    run()
