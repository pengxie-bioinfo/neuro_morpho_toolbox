from setuptools import setup

setup(name='neuro_morpho_toolbox',
      version='0.1',
      description='A toolbox to analyze neuron/brain images',
      author='Peng Xie',
      license='Allen Institute',
      packages=['neuro_morpho_toolbox'],
      install_requires=['numpy', 'sklearn', 'scipy', 'pandas', 'SimpleITK',
                        'matplotlib', 'python-igraph', 'seaborn','sparse',
			'hdbscan','colorlover', 'plotly', 
			'ipywidgets','umap-learn'
                        ],#'chart-studio'
	
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3']
      )
