# Copyright 2023 Bunting Labs, Inc.

def classFactory(interface):
    from .plugin import BuntingLabsPlugin
    return BuntingLabsPlugin(interface)
