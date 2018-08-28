#!/usr/bin/env python

'''
 This is template only, but NOT WORKING example. 
'''

import os
import testarium


@testarium.testarium.set_print
def print_web(commit):
    # support for old commits with 'cross'
    name = 'valid' if 'valid' in commit.desc else 'cross'

    # comment exceptions for unsupported characters
    try:
        comment = str(commit.desc['comment'])
        comment = comment.replace('{', '').replace('}', '').replace('"', '').replace('[', '').replace(']', '')
    except Exception as e:
        comment = str(e)

    # format print
    return ['name', 'train_loss', 'val_loss', 'time', 'comment', 'train', 'val'],  \
           [commit.name,
            '%0.3f' % (commit.desc['score']),
            '%0.3f' % (commit.desc[name]),
            '%0.0f' % (commit.desc['duration']),
            comment,
            'graph://storage/' + commit.dir + '/plot_loss.json',
            'graph://storage/' + commit.dir + '/plot_%s.json' % name,
            ]

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    testarium.testarium.best_score_is_min()
    testarium.main()
