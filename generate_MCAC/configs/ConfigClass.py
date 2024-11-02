class ConfigClass:
    def __init__(self, default_configuration, specific_configuration):
        for k, v in default_configuration.items():
            if specific_configuration is not None and k in specific_configuration:
                spec_k = specific_configuration[k]
            else:
                spec_k = None
            if type(v) is dict:
                setattr(self, k, ConfigClass(v, spec_k))
            else:
                if spec_k is None:
                    setattr(self, k, v)
                else:
                    setattr(self, k, spec_k)