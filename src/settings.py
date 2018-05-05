# -*- coding: utf-8 -*-
# Settings.py : Load Settings from json config file

import json


class Settings:

    LOADED_FILE = None
    CURRENT_STEP = 0

    @staticmethod
    def add_attr(name, value):
        """ Statically adds a parameter as an attribute
        to class Settings. All new Settings attributes
        are in capital letters.

        :param name: str
            Name of the new attribute
        :param value: object
            Value of the corresponding hyper-parameter
        """
        name = name.upper()
        setattr(Settings, name, value)

    @staticmethod
    def get_attr(name):
        return getattr(Settings, name.upper())

    @staticmethod
    def load(filepath):
        """ Statically loads the hyper-Settings from a json file

        :param filepath: str
            Path to the json parameter file
        """
        Settings.LOADED_FILE = filepath
        Settings.TO_UPDATE = list()
        with open(filepath, "r") as f:
            data = json.load(f)
            for key in sorted(data.keys()):
                if isinstance(data[key], dict):
                    Settings.add_attr(key, data[key]["value"])
                    
    @staticmethod
    def update():
        if Settings.LOADED_FILE is None:
            print('[Warning] Trying to save Settings but none have been loaded.')
            return
        with open(Settings.LOADED_FILE, "r") as f:
            data = json.load(f)
            for key in data:
                if not isinstance(data[key], dict) or key not in Settings.TO_UPDATE:
                    continue
                if data[key]['value'] != Settings.get_attr(key):
                    data[key]["value"] = Settings.get_attr(key)
        with open(Settings.LOADED_FILE, "w") as f:
            pretty_str = json.dumps(data, indent=4, sort_keys=True)
            f.write(pretty_str)
