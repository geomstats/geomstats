project = 'Geomstats'
copyright = '2019, Geomstats, Inc.'
author = 'Geomstats Team'

version = '0.1'
# The full version, including alpha/beta/rc tags
release = '0.1'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon', # enable numpy or google docstrings
]

# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

pygments_style = None

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

htmlhelp_basename = 'geomstatsdoc'

latex_elements = {
 }


latex_documents = [
    (master_doc, 'geomstats.tex', 'geomstats Documentation',
     'Geomstats Team', 'manual'),
]

man_pages = [
    (master_doc, 'geomstats', 'geomstats Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'geomstats', 'geomstats Documentation',
     author, 'geomstats', 'One line description of project.',
     'Miscellaneous'),
]

epub_title = project
epub_exclude_files = ['search.html']
