# For visdom visualization
import visdom
import numpy as np


# Don't forget to start visdom server
vis = visdom.Visdom()
vis.text('Hello, World!')
vis.image(np.ones((3, 3)))
properties = [
    {'type': 'text', 'name': 'Text input', 'value': 'initial'},
    {'type': 'number', 'name': 'Number input', 'value': '12'},
    {'type': 'button', 'name': 'Button', 'value': 'Start'},
    {'type': 'checkbox', 'name': 'Checkbox', 'value': True},
    {'type': 'select', 'name': 'Select', 'value': 1, 'values': ['Red', 'Green', 'Blue']},
]
vis.properties(properties)
