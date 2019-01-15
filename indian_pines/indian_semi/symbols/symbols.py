#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:55:48 2018
@author: bdus

classifiction nets

结构仿照https://github.com/dmlc/gluon-cv/blob/11eb654e938b32fd746ec5f72e09a44f35e99c7a/gluoncv/model_zoo/model_zoo.py

"""
from simple import *
#from .finetuned import *

__all__ = ['get_model', 'get_model_list']

_models = {
    'simple0':simple0,
    'simple1':simple1,
    'simple2':simple2#,
    #'mobilenet0.25':ft_mobilenet0_25    
    }

def get_model(name, **kwargs):
    """Returns a pre-defined model by name
    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def get_model_list():
    """Get the entire list of model names in model_zoo.
    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return _models.keys()
