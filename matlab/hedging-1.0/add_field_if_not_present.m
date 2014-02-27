function options = add_field_if_not_present(options, field_name, default_value)
  if ~isfield(options, field_name)
    options.(field_name) = default_value;
  end
end
