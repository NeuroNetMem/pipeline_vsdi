:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

method

.. auto{{ objtype }}:: {{ fullname | replace("pipeline_vsdi.", "pipeline_vsdi::") }}

{# In the fullname (e.g. `numpy.ma.MaskedArray.methodname`), the module name
is ambiguous. Using a `::` separator (e.g. `numpy::ma.MaskedArray.methodname`)
specifies `pipeline_vsdi` as the module name. #}
