{{! Geomstats Docstring Template }}
{{summaryPlaceholder}}.

{{extendedSummaryPlaceholder}}.

{{#parametersExist}}
Parameters
----------
{{#args}}
    {{var}} : {{typePlaceholder}}
         {{descriptionPlaceholder}}.
{{/args}}
{{#kwargs}}
    {{var}} : {{typePlaceholder}}
        {{descriptionPlaceholder}}.
        Optional, default {{&default}}.
{{/kwargs}}
{{/parametersExist}}

{{#returnsExist}}
Returns
-------
{{#returns}}
    {{typePlaceholder}}: {{descriptionPlaceholder}}.
{{/returns}}
{{/returnsExist}}


{{#yieldsExist}}
Yields
-------
{{#yields}}
{{typePlaceholder}}
    {{descriptionPlaceholder}}.
{{/yields}}
{{/yieldsExist}}

{{#exceptionsExist}}
Raises
------
{{#exceptions}}
{{type}}
    {{descriptionPlaceholder}}.
{{/exceptions}}
{{/exceptionsExist}}
