"""
   Copyright © 2018 Uncharted Software Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import typing
import os

import pandas as pd  # type: ignore
import numpy as np

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils

from d3m.primitives.data_cleaning.column_type_profiler import Simon
from d3m.primitives.data_transformation.array_concatenation import FuzzyJoin
import logging
import itertools
logging.basicConfig(level=logging.DEBUG)

__all__ = ('Join',)

Inputs = container.Dataset
Outputs = container.Dataset

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

class Hyperparams(hyperparams.Hyperparams):
    accuracy = hyperparams.Hyperparameter[float](
        default=0.5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Required accuracy of join ranging from 0.0 to 1.0, where 1.0 is an exact match.'
    )
    sample_size = hyperparams.Hyperparameter[float](
        default=0.1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Sample size for evaluating efficacy of joins, expressed as percentage of size of dataset. \
            Therefore, it ranges from 0.0 to 1.0, where 1.0 would sample the whole dataset.'
    )
    greedy_search = hyperparams.UniformBool(
        default = True, 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="whether to check all semantic types for best column match, or return first column match \
            above threshold"
    )
    threshold = hyperparams.Hyperparameter[float](
        default=0.5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Cut-off for which to return a good join candidate. Expressed as fraction of sampled data set. \
            Therefore, it ranges from 0.0 to 1.0, where 1.0 be all of the rows in the sampled dataset.'
    )


class Join(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Place holder  join primitive
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '04fd720e-57e7-4b32-a556-000cb9180a0d',
            'version': __version__,
            'name': 'Join Primitive',
            ## TODO: change python path
            'python_path': 'd3m.primitives.data_transformation.array_concatenation.Join',
            'keywords': ['join', 'columns', 'dataframe'],
            'source': {
                'name': __author__,
                'contact': __contact__,
                'uris': [
                    # Unstructured URIs.
                    "https://github.com/NewKnowledge/d3m-join",
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/NewKnowledge/d3m-join.git@' +
                               '{git_commit}#egg=D3MJoin'
                               .format(git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    _FIRST_ORDER_STRING_TYPES = set(('http://schema.org/addressCountry',
                        'http://schema.org/Country',
                        'http://schema.org/City',
                        'http://schema.org/State',
                        'http://schema.org/address',
                        'https://metadata.datadrivendiscovery.org/types/FileName',
                        'http://schema.org/email'))
    _FIRST_ORDER_NUMERIC_TYPES = set(('http://schema.org/longitude',
                        'http://schema.org/latitude',
                        'http://schema.org/postalCode',
                        'https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber',
                        'http://schema.org/DateTime'))    
    _SECOND_ORDER_STRING_TYPES = set(('https://metadata.datadrivendiscovery.org/types/CategoricalData',
                        'http://schema.org/Text',
                        'http://schema.org/Boolean'))
    _SECOND_ORDER_NUMERIC_TYPES = set(('http://schema.org/Integer',
                        'http://schema.org/Float'))

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        self.random_seed = random_seed
        self.volumes = volumes

    def produce(self, *,
                left: Inputs,  # type: ignore
                right: Inputs,  # type: ignore
                timeout: float = None,
                iterations: int = None) -> base.CallResult[Outputs]:

        # attempt to extract the main table
        try:
            left_resource_id, left_df = utils.get_tabular_resource(left, None)
        except ValueError as error:
            raise exceptions.InvalidArgumentValueError("Failure to find tabular resource in left dataset") from error

        try:
            right_resource_id, right_df = utils.get_tabular_resource(right, None)
        except ValueError as error:
            raise exceptions.InvalidArgumentValueError("Failure to find tabular resource in right dataset") from error

        accuracy = self.hyperparams['accuracy']
        if accuracy <= 0.0 or accuracy > 1.0:
            raise exceptions.InvalidArgumentValueError('accuracy of ' + str(accuracy) + ' is out of range')

        sample_size = self.hyperparams['sample_size']
        if sample_size <= 0.0 or sample_size > 1.0:
            raise exceptions.InvalidArgumentValueError('sample size of ' + str(sample_size) + ' is out of range')

        # use SIMON to classify columns of both datasets according to semantic type
        hyperparams_class = Simon.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        simon_client = Simon(hyperparams=hyperparams_class.defaults(), volumes = self.volumes)
        
        logging.debug('Computing semantic types of input datasets with SIMON')
        semantic_types_left = simon_client.produce(inputs = left_df).value
        semantic_types_right = simon_client.produce(inputs = right_df).value

        logging.debug('Sampling datasets for faster join computations')
        semantic_types_left = semantic_types_left.sample(frac=self.hyperparams['sample_size'], random_state = self.random_seed)
        semantic_types_right = semantic_types_right.sample(frac=self.hyperparams['sample_size'], random_state = self.random_seed)

        logging.debug('Checking for first order semantic types matches')
        result = self._compare_results( \
            self._evaluate_semantic_types(semantic_types_left, semantic_types_right, self._FIRST_ORDER_STRING_TYPES),
            self._evaluate_semantic_types(semantic_types_left, semantic_types_right, self._FIRST_ORDER_NUMERIC_TYPES))
        if result is None:
            logging.debug('Checking for second order semantic types matches')
            result = self._compare_results( \
                self._evaluate_semantic_types(semantic_types_left, semantic_types_right, self._SECOND_ORDER_STRING_TYPES),
                self._evaluate_semantic_types(semantic_types_left, semantic_types_right, self._SECOND_ORDER_NUMERIC_TYPES))
        return result

    @classmethod
    def _evaluate_semantic_types(cls,
                                 semantic_types_left: container.Dataset,
                                 semantic_types_right: container.Dataset, 
                                 first_order_types: typing.Set[str],
                                 second_order_types: typing.Set[str] = None) -> typing.Tuple[str, str, float]:
        fuzzy_join_hyperparams_class = FuzzyJoin.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        best_match = cls.hyperparams['threshold']
        best_left_col = None
        best_right_col = None

        if second_order_types is None:
            search_types = [(val, val) for val in first_order_types]
        else:
            search_types = [(val1, val2) for val1 in first_order_types for val2 in second_order_types]
        for val1, val2 in search_types:
            logging.debug('Checking for match on semantic types {} and {}'.format(val1, val2))
            left_types = semantic_types_left.metadata.get_columns_with_semantic_type(val1)
            right_types = semantic_types_right.metadata.get_columns_with_semantic_type(val2)
            matches = [match for match in itertools.product(left_types, right_types)]
            if len(matches) > 0:
                logging.debug('Found {} matches on semantic types {} and {}'.format(len(matches), val1, val2))
                for match in matches:
                    logging.debug('Attempting fuzzy join on match of column {} from df1 and column {} from df2'.format(match[0], match[1]))
                    left_col = list(semantic_types_left)[match[0]]
                    right_col = list(semantic_types_right)[match[1]]
                    fuzzy_join_hyperparams = fuzzy_join_hyperparams_class.defaults().replace(
                        {
                            'left_col': left_col,
                            'right_col': right_col,
                            'accuracy': cls.hyperparams['accuracy'],
                        }
                    )
                    fuzzy_join = FuzzyJoin(hyperparams=fuzzy_join_hyperparams)
                    result_dataset = fuzzy_join.produce(left=semantic_types_left, right=semantic_types_right).value
                    result_dataframe = result_dataset['0']

                    join_length = result_dataframe.shape[0]
                    join_percentage = join_length / semantic_types_left.shape[0]
                    logging.debug('Fuzzy join created new dataset with {} percent of records (from sampled dataset)'.format(join_percentage*100))
                    if cls.hyperparams['greedy_search']:
                        logging.debug('Found two first-order columns, {} and {} to join with greedy search'.format(left_col, right_col))
                        if join_percentage > cls.hyperparams['threshold']:
                            return(left_col, right_col, best_match)
                    else:
                        if join_percentage > best_match:
                            best_match = join_percentage
                            best_left_col = left_col
                            best_right_col = right_col
        if best_match > cls.hyperparams['threshold']:
            logging.debug('Found two first-order columns, {} and {} to join with non-greedy search'.format(best_left_col, best_right_col))
            return(best_left_col, best_right_col, best_match)
        return None

    @classmethod
    def _compare_results(cls, 
                         string_results: typing.Tuple[str, str, float] = None, 
                         numeric_results: typing.Tuple[str, str, float] = None) -> typing.Tuple[str, str]:
        col1_strings, col2_strings, best_match_strings = string_results
        col1_numeric, col2_numeric, best_match_numeric = numeric_results
        if col1_strings is None and col1_numeric is None:
            return None
        elif col1_numeric is None: 
            return (col1_strings, col2_strings)
        elif col1_strings is None:
            return (col1_numeric, col2_numeric)
        elif best_match_numeric > best_match_strings:
            return (col1_numeric, col2_numeric)
        else:
            return (col1_strings, col2_strings)

    def multi_produce(self, *,
                      produce_methods: typing.Sequence[str],
                      left: Inputs, right: Inputs,  # type: ignore
                      timeout: float = None,
                      iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._multi_produce(produce_methods=produce_methods,
                                   timeout=timeout,
                                   iterations=iterations,
                                   left=left,
                                   right=right)

    def fit_multi_produce(self, *,
                          produce_methods: typing.Sequence[str],
                          left: Inputs, right: Inputs,  # type: ignore
                          timeout: float = None,
                          iterations: int = None) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(produce_methods=produce_methods,
                                       timeout=timeout,
                                       iterations=iterations,
                                       left=left,
                                       right=right)

if __name__ == '__main__':
    _dataset_path_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test/dataset_1'))
    _dataset_path_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test/dataset_2'))

    dataframe_1 = _load_data(self._dataset_path_1)
    dataframe_2 = _load_data(self._dataset_path_2)

    hyperparams_class = Join.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    hyperparams = hyperparams_class.defaults().replace(
        {
            'sample_size': 0.1,
            'accuracy': 0.9,
        }
    )
    volumes = {}
    volumes['simon_models_1'] = '/d071106b823ab1168879651811dd03b829ab0728ba7622785bb5d3541496c45f'
    join = Join(hyperparams=hyperparams, volumes = volumes)
    col1, col2 = join.produce(left=dataframe_1, right=dataframe_2)
    print(col1)
    print(col2)

    def _load_data(dataset_path: str) -> container.DataFrame:
        dataset_doc_path = os.path.join(dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        dataframe = dataset['0']
        dataframe.metadata = dataframe.metadata.set_for_value(dataframe)

        # set the struct type
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0),
                                                       {'structural_type': int})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 1),
                                                       {'structural_type': str})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 2),
                                                       {'structural_type': float})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 3),
                                                       {'structural_type': float})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 4),
                                                       {'structural_type': str})

        # set the semantic type
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                              'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 3), 'http://schema.org/Float')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 4), 'http://schema.org/DateTime')

        # set the roles
        for i in range(1, 2):
            dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, i),
                                  'https://metadata.datadrivendiscovery.org/types/Attribute')

        # cast the dataframe to raw python types
        dataframe['d3mIndex'] = dataframe['d3mIndex'].astype(int)
        dataframe['alpha'] = dataframe['alpha'].astype(str)

        if 'bravo' in dataframe:
            dataframe['bravo'] = dataframe['bravo'].astype(float)
            dataframe['whiskey'] = dataframe['whiskey'].astype(float)
            dataframe['sierra'] = dataframe['sierra'].astype(str)

        if 'charlie' in dataframe:
            dataframe['charlie'] = dataframe['charlie'].astype(float)
            dataframe['xray'] = dataframe['xray'].astype(float)
            dataframe['tango'] = dataframe['tango'].astype(str)

        dataset['0'] = dataframe

        return dataset