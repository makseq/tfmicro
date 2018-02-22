#!/usr/bin/env python
# noispection PyPep8Naming

import os
import numpy as np
import multiprocessing
import itertools
import fnmatch
import matplotlib.pyplot as plt
import cPickle as pickle

import tfmicro.threadgen
import features


# noinspection PyUnresolvedReferences
class Data(object):
    def __init__(self, c):
        self.c = c
        self.prepare()
        #self.unit_tests()

    # some input transform (e.g.: z-norm)
    def input_transform(self, x, inplace=False):
        return None

    # prepare your data here
    def prepare(self):
        c = self.c

        self.data_dir = c['data.directory']
        self.data_cache = c['data.cache']
        self.verbose = c['data.verbose']

        self.batch_size = c['model.batch_size']
        self.num_files = c['model.num_files']
        self.timesteps = c['model.timesteps']
        self.steps_per_epoch = c['data.n_speakers'] * (c['data.n_anchor_positives'] - 1) * c['data.n_anchor_positives'] / self.batch_size
        self.validation_steps = 50 #self.steps_per_epoch

        feature_maker = features.FeatureWrapper(c)
        f, _, _ = feature_maker.signal2features(np.zeros((8000,), dtype=np.int16), rate=c['features.samplerate'])
        self.dim = f.shape[-1]

    # split data into train & validation sets
    def split(self):
        return train, test

    def find_files(self, search_dir, pattern='*.wav'):
        files_ = []
        speakers_ = []
        sessions_ = []
        for root, dirnames, filenames in os.walk(search_dir):
            for filename in fnmatch.filter(filenames, pattern):
                files_.append(os.path.join(root, filename))
                speakers_.append(os.path.basename(root))
                sessions_.append(''.join(filename.split('_')[:-1]))
        return np.array(files_), np.array(speakers_), np.array(sessions_)

    # Prepare data
    def get_data(self, data_type):
        c = self.c
        data_cache_path = os.path.join(self.data_cache, '{}_data.dat'.format(data_type))
        if not os.path.exists(self.data_cache):
            os.makedirs(self.data_cache)

        files_, speakers_, sessions_ = None, None, None
        if os.path.exists(data_cache_path):
            if self.verbose: print('Loading {} data from cache'.format(data_type))
            files_, speakers_, sessions_ = pickle.load(open(data_cache_path, 'rb'))
        else:
            if self.verbose: print('Processing {} data'.format(data_type))
            if data_type == 'train':
                files_, speakers_, sessions_ = self.find_files(os.path.join(self.data_dir, 'train_3.5s'))
            elif data_type == 'valid':
                files_, speakers_, sessions_ = self.find_files(os.path.join(self.data_dir, 'val_3.5s'))
            elif data_type == 'test':
                files_, speakers_, sessions_ = self.find_files(os.path.join(self.data_dir, 'test_3.5s'))
            else:
                raise Exception('data type')

        return files_, speakers_, sessions_

    # generator
    @tfmicro.threadgen.threadsafe_generator
    def generator(self, mode):
        random = np.random.RandomState(42)

        c = self.c
        timesteps = self.timesteps
        num_files = self.num_files
        dim = self.dim

        files, speakers, sessions = self.get_data(mode)

        pool = multiprocessing.pool.ThreadPool(1)

        unique_speakers = np.unique(speakers)

        sampled_speakers = random.permutation(unique_speakers)[:c['data.n_speakers']]

        sub_speakers = []
        sub_sessions = []
        sub_files = []
        for s in sampled_speakers:
            ind = np.where(speakers == s)[0]
            sub_speakers.append(speakers[ind])
            sub_sessions.append(sessions[ind])
            sub_files.append(files[ind])
        speakers = np.hstack(sub_speakers)
        sessions = np.hstack(sub_sessions)
        files = np.hstack(sub_files)

        data_cache_path = os.path.join(self.data_cache, '{}_features.dat'.format(mode))
        if os.path.exists(data_cache_path):
            if self.verbose: print('Loading {} features from cache'.format(mode))
            feats = pickle.load(open(data_cache_path, 'rb'))
        else:
            if self.verbose: print('Processing {} data'.format(mode))

            feature_maker = features.FeatureWrapper(c)
            feats = feature_maker.filelist2features(files, type='dict', verbose=self.verbose)
            del feats['all_features']
            pickle.dump(feats, open(data_cache_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        while 1:

            sampled_triplets = []

            for s in sampled_speakers:
                speaker_indices_ = random.permutation(np.where(speakers == s)[0])[:c['data.n_anchor_positives']]
                speaker_files = files[speaker_indices_]

                positive_pairs = [p for p in itertools.permutations(speaker_files, 2)]

                random.shuffle(positive_pairs)

                # Sample negative anchor files
                not_speaker_indices = np.random.permutation(np.where(speakers != s)[0])[:c['data.n_anchor_negatives']]
                anchor_negatives_files = files[not_speaker_indices]
                # Compose triplets

                for (a, p), n in zip(positive_pairs, anchor_negatives_files):
                    sampled_triplets.append((a, p, n))

            x = np.array(sampled_triplets)

            work = random.permutation(len(sampled_triplets))

            if c['data.shuffle']:
                random.shuffle(work)

            for j in xrange(0, len(work), self.batch_size):
                w = work[j: j + self.batch_size]

                if len(w) != self.batch_size:
                    continue

                inputs = np.zeros((len(w), num_files, timesteps, dim))

                # multi threading task
                def task(args):
                    i, k = args

                    #feature_maker = features.FeatureWrapper(c)
                    #feats = feature_maker.filelist2features(x[k], type='dict', n_processors=1, verbose=False)

                    for j, f in enumerate(x[k]):
                        inputs[i, j, :len(feats[f]), :] = feats[f]

                # pool
                pool.map(task, zip(range(len(w)), w))

                yield inputs, x[w]

    # unit testing
    def unit_tests(self, debug=False):
        # some tests here
        g = self.generator('train')
        count = 10
        for i, (feats, triplets) in g:
            if debug:
                for j, f in enumerate(feats):
                    num_plots = f.shape[0]
                    max_val = f.max()
                    min_val = f.min()
                    print(max_val, min_val)
                    fig, ax = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
                    for k, d in enumerate(f):
                        ax[k].imshow(d.T, aspect='auto', cmap='gray',
                                     vmin=min_val, vmax=max_val,
                                     origin='lower', interpolation='none')
                        ax[k].set_title(triplets[j][k])
                    plt.show()

            for t in triplets:
                a, p, n = t
                assert a.split('/')[-2] == p.split('/')[-2] and a.split('/')[-2] != n.split('/')[-2], 'Triplets name mismatch'
            count -= 1
            if count == 0:
                break


if __name__ == '__main__':
    import json
    d = Data(json.load(open('config/config.json')))
    d.unit_tests(debug=True)






