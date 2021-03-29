"""Sphinx configuration file."""

import geomstats


project = 'Geomstats'
copyright = '2019-2020, Geomstats, Inc.'
author = 'Geomstats Team'
release = version = geomstats.__version__

extensions = [
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.load_style',
]

# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

# Configure nbsphinx for notebooks execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = 'never'

# To get a prompt similar to the Classic Notebook, use
nbsphinx_input_prompt = ' In [%s]:'
nbsphinx_output_prompt = ' Out [%s]:'

nbsphinx_allow_errors = True

templates_path = ['_templates']

source_suffix = ['.rst', '.ipynb']

master_doc = 'index'

language = None

nbsphinx_thumbnails = {
    'notebooks/usecase_emg_sign_classification_in_spd_manifold':'../../notebooks'
                                                                '/figures/paper_rock_scissors.png',
    'notebooks/usecase_graph_embedding_and_clustering_in_hyperbolic_space':'../../notebooks'
                                                                            '/figures/karate_graph.png',
    'notebooks/usecase_optic_nerve_heads_analysis_in_kendall_shape_space':'../../notebooks'
                                                                            '/figures/optic_nerves.png',
    'notebooks/usecase_hand_poses_analysis_in_kendall_shape_space':'../..'
                                                                    '/notebooks/figures/hand_skeleton.png',
}

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      <p>Notebook source code:
        <a class="reference external"
        href="https://github.com/geomstats/geomstats/blob/master/\
            {{ docname|e }}">{{ docname|e }}</a>
        <br>Run it yourself on binder\
        <a href="https://mybinder.org/v2/gh/geomstats/geomstats/master?filepath=\
                            {{ docname|e }}"><img alt="Binder badge" \
                            src="https://mybinder.org/badge_logo.svg" \
                            style="vertical-align:text-bottom"></a>
      </p>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

pygments_style = None

html_theme = 'sphinx_rtd_theme'
html_baseurl = 'geomstats.github.io'
htmlhelp_basename = 'geomstatsdoc'
html_last_updated_fmt = '%c'

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
