#!/usr/bin/env python
import os
import testarium


# train run
@testarium.experiment.set_run
def run(commit):
    return 0

### Scoring
@testarium.experiment.set_score
def score(commit):
    return {}

@testarium.testarium.set_print
def print_web(commit):
    '''
    n_images = 7
    images = [''] * n_images
    count = 0
    if os.path.exists(commit.dir+'/images/'):
        for f in sorted(os.listdir(commit.dir+'/images/')):
            i = int(os.path.basename(f).split('.')[0])
            images[i] = 'image://storage/'+commit.dir+'/images/' + f
            count += 1
            if count >= n_images: break
    '''

    try: comment = str(commit.desc['comment']).replace('{','').replace('}','').replace('"','').replace('[','').replace(']','')
    except: comment = commit.desc['comment']

    h = ['name', 'train_loss', 'val_loss', 'time', 'comment', 'train_g', 'val_g']  # + ['img'+str(i) for i in xrange(n_images)]

    b = [commit.name,
        '%0.3f' % (commit.desc['score']),
        '%0.3f' % (commit.desc['cross']),
        '%0.0f'%(commit.desc['duration']),
        comment,
        'graph://storage/'+commit.dir+'/plot_loss.json',
        'graph://storage/'+commit.dir+'/plot_cross.json']  # + images

    return h, b


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    testarium.testarium.best_score_is_min()
    testarium.main()
