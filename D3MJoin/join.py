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
                               '{git_commit}#egg=d3m-join'
                               .format(git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

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
        semantic_types_left = simon_client.produce_metafeatures(inputs = left_df)
        semantic_types_right = simon_client.produce_metafeatures(inputs = right_df)

        print(semantic_types_left)
        print(semantic_types_right)

        # define some hierarchy of columns to check
            # 1. multilabel matches
            # 2. matches for interesting columns (country, city, etc.) maybe interesting cols should be supplied by user??
            # 3. matches for interesting columns with boring columns (country - text)
        
        # for each column check:
            # 1. evaluate join using sample_size + fuzzy_join primitive
            # 2. if number of records > threshold, return these columns as match, otherwise continue
            # ** how is threshold defined? HP - what should default be?

        # perform join based on semantic type
        '''
        join_type = self._get_join_semantic_type(left, left_resource_id, left_col, right, right_resource_id, right_col)
        joined: pd.Dataframe = None
        if join_type in self._STRING_JOIN_TYPES:
            joined = self._join_string_col(left_df, left_col, right_df, right_col, accuracy)
        elif join_type in self._NUMERIC_JOIN_TYPES:
            joined = self._join_numeric_col(left_df, left_col, right_df, right_col, accuracy)
        elif join_type in self._DATETIME_JOIN_TYPES:
            joined = self._join_datetime_col(left_df, left_col, right_df, right_col, accuracy)
        else:
            raise exceptions.InvalidArgumentValueError('join not surpported on type ' + str(join_type))

        # create a new dataset to hold the joined data
        resource_map = {}
        for resource_id, resource in left.items():  # type: ignore
            if resource_id == left_resource_id:
                resource_map[resource_id] = joined
            else:
                resource_map[resource_id] = resource
        result_dataset = container.Dataset(resource_map)

        return base.CallResult(result_dataset)
        '''
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
