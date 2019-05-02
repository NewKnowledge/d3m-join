from distutils.core import setup

setup(
    name='D3MJoin',
    version='0.1.0',
    description='D3M Join Primitive',
    packages=['d3m-join'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas >= 0.22.0',
        'frozendict>=1.2',
        'fuzzywuzzy>=0.17.0',
        'python-Levenshtein>=0.12.0',
        'd3m==2019.4.4'
        ],
    entry_points={
        'd3m.primitives': [
            'data_transformation.array_concatenation.FuzzyJoin = D3MJoin.fuzzy_join:FuzzyJoin',
            'data_transformation.array_concatenation.Join = D3MJoin.join:Join'
        ],
    }
)
