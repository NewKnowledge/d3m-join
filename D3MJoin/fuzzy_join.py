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
import csv
import collections
import sys

import frozendict  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
import math

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils
from fuzzywuzzy import fuzz, process
from dateutil import parser

__all__ = ('FuzzyJoin',)

Inputs = container.Dataset
Outputs = container.Dataset


class Hyperparams(hyperparams.Hyperparams):
    left_col = hyperparams.Hyperparameter[int](
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of left join column.'
    )
    right_col = hyperparams.Hyperparameter[int](
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of right join column.'
    )
    accuracy = hyperparams.Hyperparameter[float](
        default=0.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Requierd accuracy of join ranging from 0.0 to 1.0, where 1.0 is an exact match.'
    )


class FuzzyJoin(transformer.TransformerPrimitiveBase[Inputs,
                                                              Outputs,
                                                              Hyperparams]):
    """
    Place holder  join primitive
    """

    _STRING_JOIN_TYPES = set(('https://metadata.datadrivendiscovery.org/types/CategoricalData',
                              'http://schema.org/Text',
                              'http://schema.org/Boolean'))

    _NUMERIC_JOIN_TYPES = set(('http://schema.org/Integer', 'http://schema.org/Float'))

    _DATETIME_JOIN_TYPES = set(('http://schema.org/DateTime',))

    _SUPPORTED_TYPES = _STRING_JOIN_TYPES.union(_NUMERIC_JOIN_TYPES).union(_DATETIME_JOIN_TYPES)

    __author__ = 'Uncharted Software',
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '6c3188bf-322d-4f9b-bb91-68151bf1f17f',
            'version': '0.1.0',
            'name': 'Fuzzy Join Primitive',
            ## TODO: change python path
            'python_path': 'd3m.primitives.data_transformation.array_concatenation.FuzzyJoin',
            'keywords': ['join', 'columns', 'dataframe'],
            'source': {
                'name': 'Uncharted Software',
                'contact': 'mailto:cbethune@uncharted.software'
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

        left_col = self.hyperparams['left_col']
        right_col = self.hyperparams['right_col']

        # perform join based on semantic type
        join_type = self._get_join_semantic_type(left_df, left_col, right_df, right_col)
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

    @classmethod
    def _get_join_semantic_type(cls,
                                left_df: container.pandas.DataFrame,
                                left_col: int,
                                right_df: container.pandas.DataFrame,
                                right_col: int) -> typing.Optional[str]:
        # get semantic types for left and right cols
        left_types = cls._get_column_semantic_type(left_df, left_col)
        right_types = cls._get_column_semantic_type(right_df, right_col)
        
        # extract supported types
        supported_left_types = left_types.intersection(cls._SUPPORTED_TYPES)
        supported_right_types = right_types.intersection(cls._SUPPORTED_TYPES)
        
        # check for exact match
        join_types = list(supported_left_types.intersection(supported_right_types))
        if len(join_types) == 0:
            if len(left_types.intersection(cls._NUMERIC_JOIN_TYPES)) > 0 and \
                len(right_types.intersection(cls._NUMERIC_JOIN_TYPES)) > 0:
                # no exact match, but FLOAT and INT are allowed to join
                join_types = ['http://schema.org/Float']
            elif len(left_types.intersection(cls._STRING_JOIN_TYPES)) > 0 and \
                len(right_types.intersection(cls._STRING_JOIN_TYPES)) > 0:
                # no exact match, but any text-based type is allowed to join
                join_types = ['http://schema.org/Text']

        if len(join_types) > 0:
            return join_types[0]
        return None

    @classmethod
    def _get_column_semantic_type(cls,
                                  df: container.pandas.DataFrame,
                                  index: int) -> typing.Set[str]:
        col_metadata = df.metadata.query_column(index)   
        return set(col_metadata.get('semantic_types', ()))

    @classmethod
    def _string_fuzzy_match(cls,
                            match: typing.Any,
                            choices: typing.Sequence[typing.Any],
                            min_score: float) -> typing.Optional[str]:
        choice, score = process.extractOne(match, choices)
        val = None
        if score >= min_score:
            val = choice
        return val

    @classmethod
    def _join_string_col(cls,
                         left_df: container.DataFrame,
                         left_col: int,
                         right_df: container.DataFrame,
                         right_col: int,
                         accuracy: float) -> pd.DataFrame:
        # use d3mIndex from left col if present
        right_df = right_df.drop(columns='d3mIndex')
        left_col = list(left_df)[left_col]
        right_col = list(right_df)[right_col]

        # pre-compute fuzzy matches
        left_keys = left_df[left_col].unique()
        right_keys = right_df[right_col].unique()
        matches: typing.Dict[str, typing.Optional[str]] = {}
        for left_key in left_keys:
            matches[left_key] = cls._string_fuzzy_match(left_key, right_keys, accuracy*100)

        # look up pre-computed fuzzy match for each element in the left column
        left_df.index = left_df[left_col].map(lambda key: matches[key])

        # make the right col the right dataframe index
        right_df = right_df.set_index(right_col)

        # inner join on the left / right indices
        joined = container.DataFrame(left_df.join(right_df, lsuffix='_1', rsuffix='_2', how='inner'))

        # sort on the d3m index if there, otherwise use the joined column
        if 'd3mIndex' in joined:
            joined = joined.sort_values(by=['d3mIndex'])
        else:
            joined = joined.sort_values(by=[left_col])
        joined = joined.reset_index(drop=True)

        return joined

    @classmethod
    def _numeric_fuzzy_match(cls,
                             match: float,
                             choices: typing.Sequence[typing.Any],
                             accuracy: float) -> typing.Optional[float]:
        # not sure if this is faster than applying a lambda against the sequence - probably is
        inv_accuracy = 1.0 - accuracy
        min_distance = float("inf")
        min_val = None
        tolerance = float(match) * inv_accuracy
        for i, num in enumerate(choices):
            distance = abs(match - num)
            if distance <= tolerance and distance <= min_distance:
                min_diff = distance
                min_val = num
        return min_val

    @classmethod
    def _join_numeric_col(cls,
                          left_df: container.DataFrame,
                          left_col: int,
                          right_df: container.DataFrame,
                          right_col: int,
                          accuracy: float) -> pd.DataFrame:
        # use d3mIndex from left col if present
        right_df = right_df.drop(columns='d3mIndex')
        left_col = list(left_df)[left_col]
        right_col = list(right_df)[right_col]

        # fuzzy match each of the left join col against the right join col value and save the results as the left
        # dataframe index
        right_df[right_col] = pd.to_numeric(right_df[right_col])
        choices = right_df[right_col].unique()
        left_df[left_col] = pd.to_numeric(left_df[left_col])
        left_df.index = left_df[left_col]. \
            map(lambda x: cls._numeric_fuzzy_match(x, choices, accuracy))

        # make the right col the right dataframe index
        right_df = right_df.set_index(right_col)

        # inner join on the left / right indices
        joined = container.DataFrame(left_df.join(right_df, lsuffix='_1', rsuffix='_2', how='inner'))

        # sort on the d3m index if there, otherwise use the joined column
        if 'd3mIndex' in joined:
            joined = joined.sort_values(by=['d3mIndex'])
        else:
            joined = joined.sort_values(by=[left_col])
        joined = joined.reset_index(drop=True)

        return joined

    @classmethod
    def _join_datetime_col(cls,
                           left_df: container.DataFrame,
                           left_col: int,
                           right_df: container.DataFrame,
                           right_col: int,
                           accuracy: float) -> pd.DataFrame:
        # use d3mIndex from left col if present
        right_df = right_df.drop(columns='d3mIndex')
        left_col = list(left_df)[left_col]
        right_col = list(right_df)[right_col]

        # compute a tolerance delta for time matching based on a percentage of the minimum left/right time
        # range
        choices = np.array([np.datetime64(parser.parse(dt)) for dt in right_df[right_col].unique()])
        left_keys = np.array([np.datetime64(parser.parse(dt)) for dt in left_df[left_col].values])
        time_tolerance = (1.0 - accuracy) * cls._compute_time_range(left_keys, choices)
        
        left_df.index = np.array([cls._datetime_fuzzy_match(dt, choices, time_tolerance) for dt in left_keys])

        # make the right col the right dataframe index
        right_df = right_df.set_index(right_col)

        # inner join on the left / right indices
        joined = container.DataFrame(left_df.join(right_df, lsuffix='_1', rsuffix='_2', how='inner'))

        # sort on the d3m index if there, otherwise use the joined column
        if 'd3mIndex' in joined:
            joined = joined.sort_values(by=['d3mIndex'])
        else:
            joined = joined.sort_values(by=[left_col])
        joined = joined.reset_index(drop=True)

        return joined

    @classmethod
    def _datetime_fuzzy_match(cls,
                              match: np.datetime64,
                              choices: typing.Sequence[np.datetime64],
                              tolerance: np.timedelta64) -> typing.Optional[np.datetime64]:
        min_distance = None
        min_date = None
        for i, date in enumerate(choices):
            distance = abs(match - date)            
            if distance <= tolerance and (min_distance is None or distance < min_distance):
                min_distance = distance
                min_date = date
        return min_date

    @classmethod
    def _compute_time_range(cls,
                            left: typing.Sequence[np.datetime64],
                            right: typing.Sequence[np.datetime64]) -> float:
        left_min = np.amin(left)
        left_max = np.amax(left)
        left_delta = left_max - left_min

        right_min = np.amin(right)
        right_max = np.amax(right)
        right_delta = right_max - right_min

        return min(left_delta, right_delta) 
