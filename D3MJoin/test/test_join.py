import unittest
from os import path
import os
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from d3m.primitives.data_transformation.array_concatenation import Join
from d3m.metadata import base as metadata_base


class JoinPrimitiveTestCase(unittest.TestCase):

    _dataset_path_1 = path.abspath(path.join(path.dirname(__file__), 'dataset_1'))
    _dataset_path_2 = path.abspath(path.join(path.dirname(__file__), 'dataset_2'))

    def test_string_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = \
            Join.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'sample_size': 0.5,
                'accuracy': 0.9,
            }
        )
        volumes = {}
        volumes['simon_models_1'] = '/d071106b823ab1168879651811dd03b829ab0728ba7622785bb5d3541496c45f'
        join = Join(hyperparams=hyperparams, volumes = volumes)
        join.produce(left=dataframe_1, right=dataframe_2)
        #result_dataframe = result_dataset['0']

        # verify the output
        '''
        self.assertListEqual(list(result_dataframe), ['d3mIndex', 'alpha', 'bravo', 'whiskey', 
                                                      'sierra', 'charlie', 'xray', 'tango'])
        self.assertListEqual(list(result_dataframe['d3mIndex']), [1, 2, 3, 4, 5, 7, 8])
        self.assertListEqual(list(result_dataframe['alpha']),
                             ['yankee', 'yankeee', 'yank', 'Hotel', 'hotel', 'foxtrot aa', 'foxtrot'])
        self.assertListEqual(list(result_dataframe['bravo']), [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0])
        self.assertListEqual(list(result_dataframe['charlie']),
                             [100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0])
        '''
    def _load_data(cls, dataset_path: str) -> container.DataFrame:
        dataset_doc_path = path.join(dataset_path, 'datasetDoc.json')

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
            add_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                              'http://schema.org/Integer')
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


if __name__ == '__main__':
    unittest.main()
