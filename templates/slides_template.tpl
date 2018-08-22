{%- extends 'slides_reveal.tpl' -%}

{% block input_group -%}
<div class="input_hidden">
{{ super() }}
</div>
{% endblock input_group %}

{% block output_area_prompt %}
    <div class="prompt"> </div>
{% endblock output_area_prompt %}

{%- block header -%}
{{ super() }}

<style type="text/css">
//div.output_wrapper {
//  margin-top: 0px;
//}
.input_hidden {
  display: none;
//  margin-top: 5px;
}
</style>

{%- endblock header -%}
